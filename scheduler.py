"""Планировщик очереди генерации.

Вынесен из main.py и НЕ тянет тяжёлых зависимостей (torch, diffusers, whisperx),
чтобы логику планирования можно было покрыть юнит-тестами где угодно.

Политика в три уровня, сверху вниз:

  * **Допуск.** Больше ``max_user_inflight`` задач одного человека в работе не
    берём — :meth:`Scheduler.enqueue` отказывает. Это единственное, что защищает
    карту от одного увлёкшегося пользователя.
  * **Справедливость.** Люди обслуживаются по кругу, по ``max_user_batch`` задач
    подряд. Один в очереди — идут все его задачи подряд, круг из одного; пришёл
    второй — дальше чередуются.
  * **Эффективность.** Внутри выбранного человека работают прежние правила:
    добиваем тёплый видео-подтип (чтобы не перегружать ~50 ГБ) и не мариновать
    лёгкие задачи.

Анти-старвейшн ``max_wait_secs`` живёт ВНУТРИ третьего уровня: забытую задачу
выбранного человека обслуживаем вне батчинга. Поверх круга его ставить нельзя —
при глубокой очереди просрочены оказываются все задачи разом, и правило
вырождается в FIFO, отменяя справедливость. Между людьми от голодания защищает
сам круг: каждый получает свой квант за один оборот.

Задача — это dict как минимум с "type", "ts" (unix-время постановки) и "user"
(id владельца; None — общий анонимный пользователь, который в круге и под
потолком участвует наравне с остальными).
"""
import threading
import time


def user_of(job):
    """Владелец задачи.

    Задачи без владельца делят одного общего анонимного пользователя: и место
    в круге, и потолок у них общие, как у любого другого человека.
    """
    return job.get("user")


class Policy:
    """Состояние политики: что тёплое на карте и чей сейчас черёд.

    Отдельным объектом, потому что нужно дважды: планировщику — для настоящего
    выбора, снимку — для прогона будущего порядка на копии.
    """

    __slots__ = ("resident_vtype", "subtype_streak", "video_streak",
                 "user_order", "current_user", "user_streak")

    def __init__(self):
        self.resident_vtype = None   # видео-подтип, сейчас загруженный в слот
        self.subtype_streak = 0      # видео этого подтипа обслужено подряд
        self.video_streak = 0        # видео любого типа обслужено подряд
        self.user_order = []         # круг: порядок, в котором доходит очередь
        self.current_user = None     # кого обслуживаем прямо сейчас
        self.user_streak = 0         # сколько его задач обслужено подряд

    def copy(self):
        other = Policy()
        for name in self.__slots__:
            setattr(other, name, getattr(self, name))
        other.user_order = list(self.user_order)

        return other

    def record(self, job, video_types):
        """Учитывает обслуженную задачу."""
        user = user_of(job)
        if user == self.current_user:
            self.user_streak += 1
        else:
            # сменили человека — прежний уходит в конец круга, чтобы в
            # следующий раз до него дошло не раньше, чем до остальных
            if self.current_user in self.user_order:
                self.user_order.remove(self.current_user)
                self.user_order.append(self.current_user)
            self.current_user = user
            self.user_streak = 1

        jtype = job["type"]
        if jtype in video_types:
            self.video_streak += 1
            if jtype == self.resident_vtype:
                self.subtype_streak += 1
            else:
                self.resident_vtype = jtype
                self.subtype_streak = 1
        else:
            # лёгкая задача сбрасывает счётчик видео подряд; видео-слот не
            # трогаем, поэтому resident_vtype/subtype_streak сохраняются
            self.video_streak = 0


def pick_user(pending, policy, *, max_user_batch):
    """Чей сейчас черёд. Возвращает id пользователя, у которого есть задачи."""
    waiting = {user_of(j) for j in pending}

    # квант не исчерпан и у текущего ещё есть задачи — продолжаем его
    if policy.current_user in waiting and policy.user_streak < max_user_batch:
        return policy.current_user

    # круг: первый по порядку ротации, у кого есть что обслуживать
    for user in policy.user_order:
        if user in waiting and user != policy.current_user:
            return user

    # больше никого — текущий один в очереди, отдаём ему всё подряд
    if policy.current_user in waiting:
        return policy.current_user

    # круг не знает об этих задачах (например, планировщик только поднялся)
    return user_of(min(pending, key=lambda j: j["ts"]))


def pick_job(pending, video_types, policy, *, max_video_batch, max_wait_secs,
             max_videos_before_cheap, max_user_batch, now=None):
    """Чистая функция выбора следующей задачи из непустого ``pending``.

    Возвращает выбранный элемент ``pending`` (не удаляя его).
    """
    if now is None:
        now = time.time()

    user = pick_user(pending, policy, max_user_batch=max_user_batch)
    mine = [j for j in pending if user_of(j) == user]

    # Анти-старвейшн — ВНУТРИ выбранного человека, а не поверх круга.
    #
    # Поверх круга он вредит: при глубокой очереди просроченными становятся
    # сразу все задачи, правило вырождается в чистый FIFO и отменяет
    # справедливость ровно тогда, когда она нужнее всего. Замерено на
    # симуляции: с 50 жадными соседями чужая задача ждала 251-ю позицию с
    # глобальным правилом и 101-ю без него. Между людьми от голодания защищает
    # сам круг, а это правило спасает забытую задачу внутри одного человека —
    # например, старый i2v за длинной серией t2v.
    oldest = min(mine, key=lambda j: j["ts"])
    if now - oldest["ts"] >= max_wait_secs:
        return oldest

    # не мариновать лёгкие задачи: после N видео подряд пропускаем вперёд
    # ожидающую картинку/транскрипцию (видео-слот при этом остаётся тёплым)
    cheap = [j for j in mine if j["type"] not in video_types]
    if cheap and policy.video_streak >= max_videos_before_cheap:
        return min(cheap, key=lambda j: j["ts"])

    # держим видео-модель тёплой: добиваем задачи резидентного видео-подтипа
    if policy.resident_vtype in video_types and policy.subtype_streak < max_video_batch:
        same = [j for j in mine if j["type"] == policy.resident_vtype]
        if same:
            return min(same, key=lambda j: j["ts"])

    return min(mine, key=lambda j: j["ts"])


class Scheduler:
    """Потокобезопасная очередь с допуском, кругом по людям и батчингом.

    Продюсеры зовут :meth:`enqueue`; единственный потребитель крутит
    :meth:`next_job` (блокирующе) и обязан позвать :meth:`finish`, когда задача
    досчитана, — иначе место в допуске за пользователем останется занятым
    навсегда.
    """

    def __init__(self, video_types, *, max_video_batch=10, max_wait_secs=900,
                 max_videos_before_cheap=3, max_user_batch=2, max_user_inflight=5):
        self.video_types = tuple(video_types)
        self.max_video_batch = max_video_batch
        self.max_wait_secs = max_wait_secs
        self.max_videos_before_cheap = max_videos_before_cheap
        self.max_user_batch = max_user_batch
        self.max_user_inflight = max_user_inflight

        self._pending = []
        self._inflight = {}   # пользователь -> задач в работе (в очереди + на счёте)
        self._cv = threading.Condition()
        self._stopping = False

        # состояние политики (публичное — удобно смотреть и проверять в тестах)
        self.policy = Policy()

    # ------------------------------------------------------------------
    #  Снаружи и в тестах эти три читаются как поля планировщика.
    # ------------------------------------------------------------------
    @property
    def resident_vtype(self):
        return self.policy.resident_vtype

    @property
    def subtype_streak(self):
        return self.policy.subtype_streak

    @property
    def video_streak(self):
        return self.policy.video_streak

    def enqueue(self, job):
        """Ставит задачу в очередь.

        Возвращает False, ничего не поставив, если у пользователя уже
        ``max_user_inflight`` задач в работе.

        Задачи без владельца — это один общий анонимный пользователь: и потолок,
        и круг у них общие. Иначе безымянными задачами потолок обходился бы.
        """
        job.setdefault("ts", time.time())
        user = user_of(job)

        with self._cv:
            if self._inflight.get(user, 0) >= self.max_user_inflight:
                return False

            self._inflight[user] = self._inflight.get(user, 0) + 1
            if user not in self.policy.user_order:
                self.policy.user_order.append(user)
            self._pending.append(job)
            self._cv.notify()

        return True

    def finish(self, job):
        """Задача досчитана: освобождает место в допуске под следующую."""
        user = user_of(job)

        with self._cv:
            left = self._inflight.get(user, 0) - 1
            if left > 0:
                self._inflight[user] = left
            else:
                self._inflight.pop(user, None)
                self._prune_order()

    def inflight_count(self, user):
        """Сколько задач этого пользователя сейчас в работе."""
        with self._cv:
            return self._inflight.get(user, 0)

    def stop(self):
        with self._cv:
            self._stopping = True
            self._cv.notify_all()

    def pending_count(self):
        with self._cv:
            return len(self._pending)

    def _prune_order(self):
        """Выкидывает из круга тех, у кого не осталось задач.

        Иначе список растёт с каждым новым пользователем и не сокращается
        никогда. Вернувшийся встанет в конец круга — он только что
        обслуживался, ждать ему не обиднее прочих.
        """
        alive = {user_of(j) for j in self._pending} | set(self._inflight)
        self.policy.user_order = [u for u in self.policy.user_order if u in alive]

    def _pick(self, now=None):
        """Выбор без удаления и без учёта (вызывать под ``self._cv``)."""
        return pick_job(
            self._pending, self.video_types, self.policy,
            max_video_batch=self.max_video_batch,
            max_wait_secs=self.max_wait_secs,
            max_videos_before_cheap=self.max_videos_before_cheap,
            max_user_batch=self.max_user_batch,
            now=now,
        )

    def service_order(self, now=None):
        """Порядок, в котором очередь будет разобрана, — прогон на копии.

        Настоящий порядок не FIFO: его задают и круг по людям, и батчинг, так
        что по одному списку ожидающих его не восстановить. Прогон нужен, чтобы
        снаружи (в боте) можно было честно сказать «ты N-й», а не прикидывать
        по времени постановки.
        """
        with self._cv:
            pending = list(self._pending)
            policy = self.policy.copy()

        order = []
        while pending:
            job = pick_job(
                pending, self.video_types, policy,
                max_video_batch=self.max_video_batch,
                max_wait_secs=self.max_wait_secs,
                max_videos_before_cheap=self.max_videos_before_cheap,
                max_user_batch=self.max_user_batch,
                now=now,
            )
            pending.remove(job)
            policy.record(job, self.video_types)
            order.append(job)

        return order

    def snapshot(self, now=None):
        """Согласованный слепок состояния для диагностики (``/api/queue``).

        ``data`` задач намеренно не отдаём: у i2v там сырые байты картинки.
        ``pending`` идёт в том порядке, в котором очередь будет разобрана.
        """
        order = self.service_order(now=now)

        with self._cv:
            return {
                "pending": [{"id": j.get("id"), "type": j["type"], "ts": j["ts"],
                             "user": user_of(j)} for j in order],
                "resident_vtype": self.policy.resident_vtype,
                "subtype_streak": self.policy.subtype_streak,
                "video_streak": self.policy.video_streak,
                "current_user": self.policy.current_user,
                "user_streak": self.policy.user_streak,
                "inflight": dict(self._inflight),
                "next_id": order[0].get("id") if order else None,
            }

    def _take(self, now=None):
        """Небл. ядро: выбирает, удаляет и учитывает задачу. None, если пусто.

        Общая основа для :meth:`next_job` и для тестов (там передают ``now``).
        Вызывающий обеспечивает отсутствие гонок (держит ``self._cv`` либо
        работает однопоточно).
        """
        if not self._pending:
            return None

        job = self._pick(now=now)
        self._pending.remove(job)
        self.policy.record(job, self.video_types)

        return job

    def next_job(self):
        """Блокирующе ждёт и возвращает следующую задачу; None при остановке."""
        with self._cv:
            while not self._pending and not self._stopping:
                self._cv.wait()
            if self._stopping and not self._pending:
                return None

            return self._take()
