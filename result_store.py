"""Готовые результаты на диске, чтобы перезапуск не съедал чужую работу.

Раньше они жили только в словаре в памяти, и перезапуск молча выбрасывал всё:
и досчитанное, но не забранное, и стоявшее в очереди. Человек три минуты ждал
ролик, ролик посчитался — и исчез.

**Почему SQLite, а не дамп в файл на выходе.** Перезапуск бывает не только
вежливый: OOM-killer, `docker kill`, падение процесса. При них никакой
`atexit` не сработает — то есть не сработает ровно в тех случаях, ради которых
всё и затевается. WAL фиксирует запись сразу и переживает `kill -9`.

**Почему отдельный файл, а не рядом со статистикой.** Разные жизненные циклы:
статистика долгоживущая и выгружается наружу, результаты — блобы по мегабайту
на считанные часы. Смешав их, пришлось бы таскать мегабайты при каждом вакууме
статистики.

**Почему выдача НЕ удаляет запись.** Удаление при первом чтении — это та самая
дырка, из-за которой оборвавшийся ответ терял готовую работу. Запись помечается
забранной и живёт до TTL: повторный запрос получит своё. Рост при этом ограничен
не поведением клиента, а временем, и это предсказуемо.
"""

import os
import sqlite3
import threading
import time

DB_PATH = os.getenv("RESULT_DB", "/root/results.db")
# Сколько держать результат. Замер: около мегабайта на задачу, так что даже при
# двух тысячах задач в сутки шесть часов — это полгигабайта. Хранить всё вечно
# нельзя: те же две тысячи в сутки дают 730 ГБ в год.
TTL_SECONDS = int(float(os.getenv("RESULT_TTL_HOURS", "6")) * 3600)
SWEEP_INTERVAL = int(os.getenv("RESULT_SWEEP_INTERVAL", "600"))

RESTART_MESSAGE = ("сервис перезапускался, пока задача считалась — "
                   "повторите запрос")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS results (
    id           TEXT PRIMARY KEY,
    status       TEXT NOT NULL,
    payload      BLOB,
    -- payload у готовой задачи это base64-байты, у упавшей — текст ошибки.
    -- Флаг нужен, чтобы отдать их обратно тем же типом, каким они были в
    -- памяти: бот ждёт от data строку в одном случае и base64 в другом.
    payload_text INTEGER NOT NULL DEFAULT 0,
    created_at   REAL NOT NULL,
    updated_at   REAL NOT NULL,
    collected_at REAL
);
CREATE INDEX IF NOT EXISTS results_updated ON results (updated_at);
"""

_lock = threading.Lock()
_conn = None


def _log(what, exc=None):
    suffix = f": {type(exc).__name__}: {exc}" if exc is not None else ""
    print(f"[results] {what}{suffix}", flush=True)


def init():
    """Открывает базу. При беде возвращает False — сервис работает как раньше,
    просто без переживания перезапуска."""
    global _conn
    try:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.executescript(_SCHEMA)
        conn.commit()
        _conn = conn
        _log(f"база {DB_PATH}, TTL {TTL_SECONDS // 3600} ч")
        return True
    except Exception as e:
        _log("не смог открыть базу, результаты не переживут перезапуск", e)
        _conn = None
        return False


def available():
    """Поднялось ли хранилище. Если нет — вызывающий держит нагрузку в
    памяти, как было до персистентности, и сервис просто теряет её при
    перезапуске вместо того, чтобы терять её молча и всегда."""
    return _conn is not None


def put(job_id, status, payload=None):
    """Кладёт или обновляет результат. Никогда не бросает."""
    if _conn is None:
        return
    try:
        is_text = isinstance(payload, str)
        blob = payload.encode("utf-8") if is_text else payload
        now = time.time()
        with _lock:
            _conn.execute(
                "INSERT INTO results (id, status, payload, payload_text,"
                "                     created_at, updated_at)"
                " VALUES (?, ?, ?, ?, ?, ?)"
                " ON CONFLICT(id) DO UPDATE SET"
                "   status = excluded.status,"
                "   payload = excluded.payload,"
                "   payload_text = excluded.payload_text,"
                "   updated_at = excluded.updated_at",
                (job_id, status, blob, int(is_text), now, now))
            _conn.commit()
    except Exception as e:
        _log(f"не записал результат {job_id}", e)


def fetch_payload(job_id):
    """Полезная нагрузка. Нет записи или нагрузки — None."""
    if _conn is None:
        return None
    try:
        with _lock:
            row = _conn.execute(
                "SELECT payload, payload_text FROM results WHERE id = ?",
                (job_id,)).fetchone()
    except Exception as e:
        _log(f"не прочитал результат {job_id}", e)
        return None

    if row is None or row[0] is None:
        return None
    return row[0].decode("utf-8") if row[1] else row[0]


def mark_collected(job_id):
    if _conn is None:
        return
    try:
        with _lock:
            _conn.execute(
                "UPDATE results SET collected_at = ? WHERE id = ? "
                "AND collected_at IS NULL", (time.time(), job_id))
            _conn.commit()
    except Exception as e:
        _log(f"не отметил выдачу {job_id}", e)


def delete(job_id):
    """Убирает запись совсем — например когда задачу отвергли по потолку."""
    if _conn is None:
        return
    try:
        with _lock:
            _conn.execute("DELETE FROM results WHERE id = ?", (job_id,))
            _conn.commit()
    except Exception as e:
        _log(f"не удалил {job_id}", e)


def recover():
    """Поднимает состояние после перезапуска.

    Задачи, застигнутые в очереди или на карте, досчитаться уже не могут:
    очередь планировщика жила в памяти, а работа на карте оборвалась. Честно
    переводим их в ошибку с внятным текстом — это заметно лучше тишины, в
    которой человек не понимает, ждать ему или нет.

    Возвращает словарь id -> статус для тех, что ещё живы.
    """
    if _conn is None:
        return {}, 0
    try:
        with _lock:
            cur = _conn.execute(
                "UPDATE results SET status = 'error', payload = ?,"
                " payload_text = 1, updated_at = ?"
                " WHERE status IN ('pending', 'in_progress')",
                (RESTART_MESSAGE.encode("utf-8"), time.time()))
            interrupted = cur.rowcount
            rows = _conn.execute("SELECT id, status FROM results").fetchall()
            _conn.commit()
    except Exception as e:
        _log("не поднял состояние", e)
        return {}, 0

    if interrupted:
        _log(f"оборвано перезапуском и помечено ошибкой: {interrupted}")
    return {job_id: status for job_id, status in rows}, interrupted


def sweep():
    """Убирает всё, что старше TTL. Возвращает список выкинутых id."""
    if _conn is None:
        return []
    cutoff = time.time() - TTL_SECONDS
    try:
        with _lock:
            rows = _conn.execute(
                "SELECT id FROM results WHERE updated_at < ?", (cutoff,)
            ).fetchall()
            if rows:
                _conn.execute("DELETE FROM results WHERE updated_at < ?",
                              (cutoff,))
                _conn.commit()
    except Exception as e:
        _log("уборка не удалась", e)
        return []
    return [r[0] for r in rows]


def size_bytes():
    """Сколько занимает хранилище — для /api/stats."""
    try:
        return os.path.getsize(DB_PATH)
    except OSError:
        return None
