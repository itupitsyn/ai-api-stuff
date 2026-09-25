"""Учёт задач: одна строка на задачу.

Почему локальная SQLite, а не запись сразу в Postgres на наблюдательной
машине: **генерация не должна от неё зависеть**. Та однажды уже тихо умерла на
десять месяцев, и никто этого не заметил. Здесь строка ложится на диск рядом с
сервисом, а экспортёр отвозит её когда сможет; недоступна — доедет позже.

Файл базы лежит в каталоге, смонтированном с хоста (`.:/root`), поэтому
переживает пересборку образа — в отличие от кода, который в образ запечён.

ГЛАВНОЕ ПРАВИЛО: ничто отсюда не должно ронять воркер. Статистика полезна, но
дешевле потерять строку, чем ролик. Поэтому каждая точка входа проглатывает
исключения и только пишет в лог.

Три отметки времени, а не две, — это осознанно. Долгое ОЖИДАНИЕ означает, что
не хватает карт; долгий СЧЁТ — что велик холст или тяжела модель. Слитые в одно
число, они неразличимы задним числом, и я на этом уже обжигался.
"""

import os
import sqlite3
import threading
import time
from datetime import datetime, timezone

DB_PATH = os.getenv("STATS_DB", "/root/stats.db")
# Пусто — экспортёр не поднимается, строки просто копятся локально. Это рабочий
# режим, а не деградация: локальная база и есть источник истины.
PG_DSN = os.getenv("STATS_PG_DSN", "")
EXPORT_INTERVAL = int(os.getenv("STATS_EXPORT_INTERVAL", "30"))
EXPORT_BATCH = 500
NODE = os.getenv("NODE_NAME") or os.uname().nodename

# Порядок колонок задан один раз и используется и при вставке, и при экспорте:
# разъехавшись, они молча перепутали бы значения местами.
COLUMNS = (
    "job_id", "user_id", "kind", "model",
    "width", "height", "frames", "fps",
    # source_prompt и style заполняет бот; до его доработки остаются пустыми
    "source_prompt", "prompt", "style",
    "enqueued_at", "started_at", "finished_at", "waited_s", "duration_s",
    "node", "card", "status", "error_class", "out_bytes",
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id        TEXT PRIMARY KEY,
    user_id       INTEGER,
    kind          TEXT NOT NULL,
    model         TEXT,
    width         INTEGER,
    height        INTEGER,
    frames        INTEGER,
    fps           INTEGER,
    source_prompt TEXT,
    prompt        TEXT,
    style         TEXT,
    enqueued_at   REAL,
    started_at    REAL,
    finished_at   REAL,
    waited_s      REAL,
    duration_s    REAL,
    node          TEXT,
    card          INTEGER,
    status        TEXT NOT NULL,
    error_class   TEXT,
    out_bytes     INTEGER,
    exported_at   REAL
);
CREATE INDEX IF NOT EXISTS jobs_unexported ON jobs (exported_at)
    WHERE exported_at IS NULL;
CREATE INDEX IF NOT EXISTS jobs_enqueued ON jobs (enqueued_at DESC);
"""

_lock = threading.Lock()
_conn = None


def _log(what, exc):
    print(f"[stats] {what}: {type(exc).__name__}: {exc}", flush=True)


def init():
    """Открывает базу и накатывает схему. Молча переживает любую беду."""
    global _conn
    try:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        # WAL: писать будут потоки воркеров, читать — эндпоинт и экспортёр,
        # и блокировать друг друга им незачем.
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.executescript(_SCHEMA)
        conn.commit()
        _conn = conn
        print(f"[stats] база {DB_PATH}, узел {NODE}", flush=True)
    except Exception as e:
        _log("не смог открыть базу, учёт выключен", e)
        _conn = None


def record(**fields):
    """Кладёт строку про завершённую задачу. Никогда не бросает."""
    if _conn is None:
        return
    try:
        fields.setdefault("node", NODE)
        row = [fields.get(c) for c in COLUMNS]
        placeholders = ",".join("?" * len(COLUMNS))
        with _lock:
            _conn.execute(
                f"INSERT OR REPLACE INTO jobs ({','.join(COLUMNS)}) "
                f"VALUES ({placeholders})", row)
            _conn.commit()
    except Exception as e:
        _log("строка не записалась", e)


def summary(hours=24):
    """Сводка для /api/stats. При беде возвращает пустую, а не падает."""
    if _conn is None:
        return {"enabled": False}
    since = time.time() - hours * 3600
    try:
        with _lock:
            by_kind = _conn.execute(
                "SELECT kind, status, COUNT(*), "
                "       AVG(waited_s), AVG(duration_s), MAX(waited_s) "
                "FROM jobs WHERE enqueued_at >= ? GROUP BY kind, status",
                (since,)).fetchall()
            users = _conn.execute(
                "SELECT COUNT(DISTINCT user_id) FROM jobs WHERE enqueued_at >= ?",
                (since,)).fetchone()[0]
            total, pending_export = _conn.execute(
                "SELECT COUNT(*), SUM(exported_at IS NULL) FROM jobs").fetchone()
    except Exception as e:
        _log("сводка не собралась", e)
        return {"enabled": True, "error": str(e)}

    return {
        "enabled": True,
        "window_hours": hours,
        "users": users,
        "rows_total": total,
        "rows_awaiting_export": pending_export or 0,
        "by_kind": [
            {"kind": k, "status": s, "count": n,
             "waited_avg_s": round(wa or 0, 1),
             "duration_avg_s": round(da or 0, 1),
             "waited_max_s": round(wm or 0, 1)}
            for k, s, n, wa, da, wm in by_kind
        ],
    }


# --------------------------------------------------------------------------
# Экспорт на наблюдательную машину
# --------------------------------------------------------------------------

def _to_ts(value):
    """Эпоха в SQLite -> timestamptz в Postgres. None остаётся None."""
    return None if value is None else datetime.fromtimestamp(value, timezone.utc)


_TIME_COLS = {"enqueued_at", "started_at", "finished_at"}


def _export_once():
    """Отвозит пачку строк. Возвращает, сколько увезла."""
    import psycopg

    with _lock:
        rows = _conn.execute(
            f"SELECT {','.join(COLUMNS)} FROM jobs "
            f"WHERE exported_at IS NULL ORDER BY enqueued_at LIMIT {EXPORT_BATCH}"
        ).fetchall()
    if not rows:
        return 0

    idx = {c: i for i, c in enumerate(COLUMNS)}
    converted = [
        tuple(_to_ts(v) if c in _TIME_COLS else v
              for c, v in zip(COLUMNS, row))
        for row in rows
    ]
    placeholders = ",".join(["%s"] * len(COLUMNS))
    # DO NOTHING, а не DO UPDATE: строка пишется один раз по завершении задачи
    # и больше не меняется, а повтор означает лишь неудачную отметку экспорта.
    sql = (f"INSERT INTO jobs ({','.join(COLUMNS)}) VALUES ({placeholders}) "
           f"ON CONFLICT (job_id) DO NOTHING")

    with psycopg.connect(PG_DSN, connect_timeout=10) as pg:
        with pg.cursor() as cur:
            cur.executemany(sql, converted)

    now = time.time()
    ids = [(now, row[idx["job_id"]]) for row in rows]
    with _lock:
        _conn.executemany("UPDATE jobs SET exported_at = ? WHERE job_id = ?", ids)
        _conn.commit()
    return len(rows)


def _exporter():
    """Крутится вечно. Недоступность Postgres — не ошибка, а обычное дело."""
    failures = 0
    while True:
        time.sleep(EXPORT_INTERVAL)
        if _conn is None:
            continue
        try:
            sent = _export_once()
            if sent:
                print(f"[stats] отвезено строк: {sent}", flush=True)
            failures = 0
        except Exception as e:
            failures += 1
            # Шумим только на первой неудаче и дальше редко: наблюдательная
            # машина может лежать сутками, и засыпать лог генерации этим нельзя.
            if failures == 1 or failures % 60 == 0:
                _log(f"экспорт не удался (подряд: {failures})", e)


def start_exporter():
    if not PG_DSN:
        print("[stats] STATS_PG_DSN пуст — экспорт выключен, пишем только локально",
              flush=True)
        return
    t = threading.Thread(target=_exporter, name="stats-exporter", daemon=True)
    t.start()
    print(f"[stats] экспортёр запущен, интервал {EXPORT_INTERVAL}с", flush=True)
