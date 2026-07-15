"""Тесты планировщика очереди (scheduler.py).

Не тянут torch/diffusers — проверяют чистую логику батчинга и анти-старвейшна.
Запуск:  pytest test_scheduler.py -v
"""
from scheduler import Scheduler, pick_job

# Типы задач в тестах — простые строки; настоящий main.py передаёт ProcessType.
T2V, I2V, IMG, TRANS = "t2v", "i2v", "img", "trans"
VIDEO = (T2V, I2V)


def job(jtype, ts):
    return {"id": f"{jtype}-{ts}", "type": jtype, "ts": ts}


def make_scheduler(**kw):
    kw.setdefault("max_video_batch", 10)
    kw.setdefault("max_wait_secs", 10_000)   # большой — чтобы анти-старвейшн не мешал
    kw.setdefault("max_videos_before_cheap", 3)
    return Scheduler(VIDEO, **kw)


def drain(sched, now):
    """Полностью вычерпывает очередь при фиксированном ``now``, возвращает типы."""
    order = []
    while True:
        j = sched._take(now=now)
        if j is None:
            break
        order.append(j["type"])
    return order


def pick(pending, **kw):
    kw.setdefault("resident_vtype", None)
    kw.setdefault("subtype_streak", 0)
    kw.setdefault("video_streak", 0)
    kw.setdefault("max_video_batch", 10)
    kw.setdefault("max_wait_secs", 10_000)
    kw.setdefault("max_videos_before_cheap", 3)
    kw.setdefault("now", 1000)
    return pick_job(pending, VIDEO, kw.pop("resident_vtype"),
                    kw.pop("subtype_streak"), kw.pop("video_streak"), **kw)


# --------------------------------------------------------------------------
#  pick_job — чистая функция выбора
# --------------------------------------------------------------------------

def test_fifo_when_no_video_resident():
    pending = [job(IMG, 5), job(TRANS, 1), job(IMG, 3)]
    assert pick(pending)["ts"] == 1  # самый старый


def test_prefers_resident_video_subtype_over_older_other_subtype():
    # i2v старше, но модель t2v тёплая — добиваем t2v, чтобы не перегружать 50 ГБ
    pending = [job(I2V, 1), job(T2V, 2), job(T2V, 3)]
    chosen = pick(pending, resident_vtype=T2V, subtype_streak=1, video_streak=1)
    assert chosen["type"] == T2V and chosen["ts"] == 2


def test_subtype_batch_cap_lets_other_subtype_through():
    # достигли лимита подряд по подтипу — больше не держим, отдаём самый старый
    pending = [job(I2V, 1), job(T2V, 2)]
    chosen = pick(pending, resident_vtype=T2V, subtype_streak=10, video_streak=10)
    assert chosen["type"] == I2V


def test_cheap_jumps_after_video_streak_limit():
    pending = [job(T2V, 1), job(IMG, 2)]
    # 3 видео подряд уже обслужены, картинка ждёт → пропускаем её вперёд
    chosen = pick(pending, resident_vtype=T2V, subtype_streak=3, video_streak=3)
    assert chosen["type"] == IMG


def test_cheap_does_not_jump_before_video_streak_limit():
    pending = [job(T2V, 1), job(IMG, 2)]
    chosen = pick(pending, resident_vtype=T2V, subtype_streak=2, video_streak=2)
    assert chosen["type"] == T2V


def test_starvation_guard_serves_oldest_regardless():
    # картинка ждёт дольше max_wait — обслуживаем немедленно, даже в разгар видео-пачки
    pending = [job(IMG, 0), job(T2V, 50)]
    chosen = pick(pending, resident_vtype=T2V, subtype_streak=1, video_streak=1,
                  max_wait_secs=100, now=200)
    assert chosen["type"] == IMG


# --------------------------------------------------------------------------
#  Scheduler — последовательность обслуживания (pick + учёт счётчиков)
# --------------------------------------------------------------------------

def test_batching_groups_video_subtypes_together():
    s = make_scheduler()
    for j in [job(T2V, 0), job(I2V, 1), job(T2V, 2), job(I2V, 3), job(T2V, 4)]:
        s.enqueue(j)
    # вместо FIFO (t2v,i2v,t2v,i2v,t2v = 4 перезагрузки) — одна смена модели
    assert drain(s, now=100) == [T2V, T2V, T2V, I2V, I2V]


def test_no_more_than_three_videos_when_cheap_waiting():
    s = make_scheduler()
    for j in [job(T2V, 0), job(T2V, 1), job(T2V, 2), job(T2V, 3), job(T2V, 4), job(IMG, 5)]:
        s.enqueue(j)
    order = drain(s, now=100)
    # картинка вклинивается ровно после 3 видео подряд
    assert order == [T2V, T2V, T2V, IMG, T2V, T2V]
    assert order.index(IMG) == 3


def test_cheap_keeps_video_slot_warm():
    # после «пропущенной вперёд» картинки видео-модель остаётся тёплой (без свопа)
    s = make_scheduler()
    for j in [job(T2V, 0), job(T2V, 1), job(T2V, 2), job(IMG, 3), job(T2V, 4)]:
        s.enqueue(j)
    drain(s, now=100)
    assert s.resident_vtype == T2V          # слот не менялся
    assert s.subtype_streak == 4            # серия t2v не сбилась картинкой


def test_cheap_only_is_fifo():
    s = make_scheduler()
    for j in [job(IMG, 0), job(TRANS, 1), job(IMG, 2)]:
        s.enqueue(j)
    assert drain(s, now=100) == [IMG, TRANS, IMG]


def test_single_cheap_between_batches_resets_video_streak():
    # длинная серия t2v с картинками должна дробиться максимум по 3 видео
    s = make_scheduler()
    for i in range(9):
        s.enqueue(job(T2V, i))
    s.enqueue(job(IMG, 100))
    s.enqueue(job(IMG, 101))
    order = drain(s, now=1000)
    # 3 видео, картинка, 3 видео, картинка, 3 видео
    assert order == [T2V, T2V, T2V, IMG, T2V, T2V, T2V, IMG, T2V, T2V, T2V]


def test_stop_unblocks_next_job_when_empty():
    s = make_scheduler()
    s.stop()
    assert s.next_job() is None


def test_next_job_returns_and_records():
    s = make_scheduler()
    s.enqueue(job(T2V, 0))
    j = s.next_job()
    assert j["type"] == T2V
    assert s.resident_vtype == T2V
    assert s.video_streak == 1
