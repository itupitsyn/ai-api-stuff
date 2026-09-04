"""Тесты планировщика очереди (scheduler.py).

Не тянут torch/diffusers — проверяют чистую логику батчинга и анти-старвейшна.
Запуск:  pytest test_scheduler.py -v
"""
from scheduler import Policy, Scheduler, pick_job, pick_user

# Типы задач в тестах — простые строки; настоящий main.py передаёт ProcessType.
T2V, I2V, IMG, TRANS = "t2v", "i2v", "img", "trans"
VIDEO = (T2V, I2V)


def job(jtype, ts, user=None):
    return {"id": f"{jtype}-{ts}", "type": jtype, "ts": ts, "user": user}


def make_scheduler(**kw):
    kw.setdefault("max_video_batch", 10)
    kw.setdefault("max_wait_secs", 10_000)   # большой — чтобы анти-старвейшн не мешал
    kw.setdefault("max_videos_before_cheap", 3)
    kw.setdefault("max_user_batch", 2)
    kw.setdefault("max_user_inflight", 100)  # большой — чтобы допуск не мешал
    return Scheduler(VIDEO, **kw)


def policy(**kw):
    """Состояние политики с нужными полями; остальные — по умолчанию."""
    p = Policy()
    for name, value in kw.items():
        setattr(p, name, value)
    return p


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
    state = policy(resident_vtype=kw.pop("resident_vtype", None),
                   subtype_streak=kw.pop("subtype_streak", 0),
                   video_streak=kw.pop("video_streak", 0),
                   user_order=kw.pop("user_order", []),
                   current_user=kw.pop("current_user", None),
                   user_streak=kw.pop("user_streak", 0))
    kw.setdefault("max_video_batch", 10)
    kw.setdefault("max_wait_secs", 10_000)
    kw.setdefault("max_videos_before_cheap", 3)
    kw.setdefault("max_user_batch", 2)
    kw.setdefault("now", 1000)
    return pick_job(pending, VIDEO, state, **kw)


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


# --------------------------------------------------------------------------
#  snapshot — состояние для /api/queue
# --------------------------------------------------------------------------

def test_snapshot_empty_queue():
    s = make_scheduler()
    snap = s.snapshot(now=100)
    assert snap["pending"] == []
    assert snap["next_id"] is None
    assert snap["resident_vtype"] is None


def test_snapshot_lists_pending_without_payload():
    s = make_scheduler()
    # payload у i2v — сырые байты картинки, наружу он попасть не должен
    s.enqueue({"id": "a", "type": I2V, "ts": 1, "user": 7, "data": {"image": b"\xff" * 10}})
    snap = s.snapshot(now=100)
    assert snap["pending"] == [{"id": "a", "type": I2V, "ts": 1, "user": 7}]


def test_snapshot_does_not_consume_or_record():
    s = make_scheduler()
    s.enqueue(job(T2V, 0))
    s.snapshot(now=100)
    s.snapshot(now=100)
    # слепок не должен ни забирать задачу, ни двигать счётчики батчинга
    assert s.pending_count() == 1
    assert s.resident_vtype is None and s.video_streak == 0
    assert s._take(now=100)["id"] == "t2v-0"


def test_snapshot_next_id_follows_batching_not_fifo():
    s = make_scheduler()
    s.enqueue(job(T2V, 2))
    s._take(now=100)          # обслужили t2v → слот t2v тёплый
    s.enqueue(job(I2V, 1))    # ждёт дольше всех
    s.enqueue(job(T2V, 3))
    snap = s.snapshot(now=100)
    oldest = min(snap["pending"], key=lambda j: j["ts"])
    # i2v ждёт дольше, но следующим пойдёт t2v — его модель уже загружена
    assert oldest["id"] == "i2v-1"
    assert snap["next_id"] == "t2v-3"
    assert snap["resident_vtype"] == T2V and snap["subtype_streak"] == 1


def test_snapshot_next_id_matches_take():
    s = make_scheduler()
    for j in [job(T2V, 0), job(IMG, 1), job(I2V, 2)]:
        s.enqueue(j)
    predicted = s.snapshot(now=100)["next_id"]
    assert s._take(now=100)["id"] == predicted


# --------------------------------------------------------------------------
#  Допуск: потолок задач в работе на пользователя
# --------------------------------------------------------------------------

def test_admission_caps_one_user():
    s = make_scheduler(max_user_inflight=5)
    accepted = [s.enqueue(job(T2V, i, user=1)) for i in range(6)]
    assert accepted == [True] * 5 + [False]
    assert s.pending_count() == 5


def test_admission_is_per_user():
    s = make_scheduler(max_user_inflight=2)
    assert [s.enqueue(job(T2V, i, user=1)) for i in range(3)] == [True, True, False]
    # чужой потолок соседа не касается
    assert s.enqueue(job(T2V, 10, user=2)) is True


def test_finish_frees_a_slot():
    s = make_scheduler(max_user_inflight=2)
    s.enqueue(job(T2V, 0, user=1))
    s.enqueue(job(T2V, 1, user=1))
    assert s.enqueue(job(T2V, 2, user=1)) is False

    # взяли на счёт — место всё ещё занято, задача-то считается
    served = s._take(now=100)
    assert s.enqueue(job(T2V, 3, user=1)) is False

    # досчитали — место освободилось
    s.finish(served)
    assert s.inflight_count(1) == 1
    assert s.enqueue(job(T2V, 4, user=1)) is True


def test_ownerless_jobs_share_one_anonymous_user():
    # задачи без владельца — один общий человек: и потолок, и круг общие
    s = make_scheduler(max_user_inflight=2)
    assert [s.enqueue(job(T2V, i)) for i in range(3)] == [True, True, False]
    # и он не мешает названному соседу
    assert s.enqueue(job(T2V, 10, user=1)) is True


def test_anonymous_user_takes_its_turn_in_the_rotation():
    s = make_scheduler(max_user_batch=2)
    for i in range(4):
        s.enqueue(job(T2V, i))              # аноним
    for i in range(4):
        s.enqueue(job(T2V, 10 + i, user=1))
    assert users_drained(s, now=100) == [None, None, 1, 1, None, None, 1, 1]


# --------------------------------------------------------------------------
#  Справедливость: круг по людям с квантом
# --------------------------------------------------------------------------

def users_drained(sched, now):
    """Вычерпывает очередь и возвращает владельцев в порядке обслуживания."""
    order = []
    while True:
        j = sched._take(now=now)
        if j is None:
            break
        order.append(j["user"])
    return order


def test_single_user_gets_everything_in_a_row():
    # один в очереди — круг из одного, дробить нечего и не на кого
    s = make_scheduler()
    for i in range(5):
        s.enqueue(job(T2V, i, user=1))
    assert users_drained(s, now=100) == [1] * 5


def test_two_users_alternate_by_quantum():
    s = make_scheduler(max_user_batch=2)
    for i in range(5):
        s.enqueue(job(T2V, i, user=1))
    for i in range(3):
        s.enqueue(job(T2V, 10 + i, user=2))
    # по две задачи на человека, пока у обоих есть что считать
    assert users_drained(s, now=100) == [1, 1, 2, 2, 1, 1, 2, 1]


def test_three_users_rotate_evenly():
    s = make_scheduler(max_user_batch=2)
    for user in (1, 2, 3):
        for i in range(4):
            s.enqueue(job(T2V, user * 10 + i, user=user))
    assert users_drained(s, now=100) == [1, 1, 2, 2, 3, 3, 1, 1, 2, 2, 3, 3]


def test_late_user_joins_the_rotation():
    s = make_scheduler(max_user_batch=2)
    for i in range(4):
        s.enqueue(job(T2V, i, user=1))
    s._take(now=100)
    s._take(now=100)                       # квант первого исчерпан

    s.enqueue(job(T2V, 100, user=2))       # второй пришёл позже всех
    assert s._take(now=100)["user"] == 2   # и всё равно получает свой квант


def test_user_who_ran_out_leaves_the_rotation():
    s = make_scheduler(max_user_batch=2)
    s.enqueue(job(T2V, 0, user=1))
    for i in range(3):
        s.enqueue(job(T2V, 10 + i, user=2))
    # у первого одна задача; дальше круг не должен на нём спотыкаться
    assert users_drained(s, now=100) == [1, 2, 2, 2]


def test_rotation_does_not_break_video_batching():
    # внутри своего кванта человек всё так же добивает тёплый подтип
    s = make_scheduler(max_user_batch=2)
    s.enqueue(job(I2V, 0, user=1))
    s.enqueue(job(T2V, 1, user=1))
    s.enqueue(job(T2V, 2, user=1))
    s.enqueue(job(T2V, 3, user=2))
    order = drain(s, now=100)
    # i2v первым (он старше), потом круг отдаёт второму его t2v — но t2v
    # первого пользователя идёт раньше, потому что квант ещё не исчерпан
    assert order == [I2V, T2V, T2V, T2V]


def test_starvation_guard_does_not_break_the_rotation():
    # Чужая просроченная задача НЕ отменяет круг: иначе при глубокой очереди
    # просрочены оказываются все разом и справедливость исчезает.
    pending = [job(T2V, 0, user=1), job(T2V, 50, user=2)]
    chosen = pick(pending, current_user=2, user_streak=0, user_order=[2, 1],
                  max_wait_secs=100, now=200)
    assert chosen["user"] == 2


def test_starvation_guard_works_inside_the_user():
    # А вот забытую задачу самого обслуживаемого человека вытаскивает
    pending = [job(IMG, 0, user=1), job(T2V, 50, user=1)]
    chosen = pick(pending, current_user=1, user_streak=0, user_order=[1],
                  resident_vtype=T2V, subtype_streak=1, video_streak=1,
                  max_wait_secs=100, now=200)
    assert chosen["type"] == IMG


def test_pick_user_keeps_current_until_quantum_spent():
    pending = [job(T2V, 0, user=1), job(T2V, 1, user=2)]
    state = policy(user_order=[1, 2], current_user=1, user_streak=1)
    assert pick_user(pending, state, max_user_batch=2) == 1

    state.user_streak = 2
    assert pick_user(pending, state, max_user_batch=2) == 2


# --------------------------------------------------------------------------
#  service_order — предсказание порядка для /api/queue
# --------------------------------------------------------------------------

def test_snapshot_pending_is_in_service_order():
    # снимок обещает порядок, в котором очередь и правда будет разобрана
    s = make_scheduler(max_user_batch=2)
    for i in range(3):
        s.enqueue(job(T2V, i, user=1))
    for i in range(2):
        s.enqueue(job(T2V, 10 + i, user=2))
    s.enqueue(job(IMG, 20, user=1))

    predicted = [j["id"] for j in s.snapshot(now=100)["pending"]]

    actual = []
    while True:
        j = s._take(now=100)
        if j is None:
            break
        actual.append(j["id"])

    assert predicted == actual


def test_service_order_does_not_consume_or_move_policy():
    s = make_scheduler()
    s.enqueue(job(T2V, 0, user=1))
    s.enqueue(job(T2V, 1, user=2))
    s.service_order(now=100)
    s.service_order(now=100)
    # прогон идёт на копии: очередь на месте, счётчики не сдвинулись
    assert s.pending_count() == 2
    assert s.resident_vtype is None and s.policy.current_user is None


def test_snapshot_reports_inflight_per_user():
    s = make_scheduler()
    s.enqueue(job(T2V, 0, user=1))
    s.enqueue(job(T2V, 1, user=1))
    s.enqueue(job(T2V, 2, user=2))
    assert s.snapshot(now=100)["inflight"] == {1: 2, 2: 1}


def test_nobody_is_pushed_to_the_end_forever():
    """Жадные соседи не могут задвинуть чужую задачу дальше одного оборота.

    Голодание тут ловится числом: жертва обязана уехать не дальше позиции
    (соседей x квант + 1). Правило выбора уже один раз это ломало — глобальный
    анти-старвейшн при глубокой очереди вырождался в FIFO и отменял круг, и с
    50 соседями жертва уезжала на 251-ю позицию вместо 101-й. Тест держит
    границу, чтобы такое не вернулось незаметно.
    """
    neighbours, quantum, cap, job_secs = 10, 2, 5, 85.0
    limit = neighbours * quantum + 1

    s = make_scheduler(max_user_batch=quantum, max_user_inflight=cap,
                       max_wait_secs=900)
    now = 0.0
    counter = 0

    for user in range(1, neighbours + 1):
        for _ in range(cap):
            counter += 1
            s.enqueue({"id": f"g{counter}", "type": T2V, "ts": now, "user": user})
    s.enqueue({"id": "victim", "type": T2V, "ts": now, "user": 999})

    served = 0
    while True:
        j = s._take(now=now)
        assert j is not None, "очередь опустела, а жертву так и не обслужили"
        s.finish(j)
        served += 1
        now += job_secs
        if j["id"] == "victim":
            break

        # сосед немедленно ставит новую задачу взамен досчитанной
        counter += 1
        s.enqueue({"id": f"g{counter}", "type": T2V, "ts": now, "user": j["user"]})
        assert served < limit, f"жертву задвинули за круг: уже {served} задач"

    assert served == limit


def test_cheap_job_is_not_pushed_to_the_end_by_video_neighbours():
    # лёгкая задача среди чужих видео ждёт тот же оборот, не дольше
    neighbours, quantum, cap = 5, 2, 5
    s = make_scheduler(max_user_batch=quantum, max_user_inflight=cap)
    now = 0.0
    counter = 0

    for user in range(1, neighbours + 1):
        for _ in range(cap):
            counter += 1
            s.enqueue({"id": f"g{counter}", "type": T2V, "ts": now, "user": user})
    s.enqueue({"id": "victim", "type": IMG, "ts": now, "user": 999})

    served = 0
    while True:
        j = s._take(now=now)
        s.finish(j)
        served += 1
        if j["id"] == "victim":
            break
        counter += 1
        s.enqueue({"id": f"g{counter}", "type": T2V, "ts": now, "user": j["user"]})

    assert served <= neighbours * quantum + 1
