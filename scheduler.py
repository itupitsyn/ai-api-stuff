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

Карт может быть несколько. Устройства — однородный пул: роли за ними не
закреплены, а «на этой карте видео-модель уже тёплая» — не правило, а повод
предпочесть задачу без смены модели. Специализация из-за этого возникает сама,
когда выгодна, и растворяется, когда нет: на двух картах видео липнет к
прогретой, а картинки уходят на соседнюю; на четырёх при видео-нагрузке все
четыре поднимут видео-модель. Поведение вытекает из чисел, а не из захардкоженных
ролей, поэтому переезд на другое железо кода не требует.

Что глобально, а что на карту:

  * круг по людям — **глобальный**: человеку важно, сколько он получил всего, а
    не на какой карте;
  * тёплая модель и счётчики батчинга — **свои у каждой карты** (:class:`Device`).

``max_concurrent_video`` ограничивает, сколько видео считается одновременно на
всём пуле. Это про оперативную память, а не про карты: каждый ComfyUI держит
свои staging-буферы (замер на боксе — около 40 ГБ), и два видео разом в 78 ГБ
не помещаются, сколько бы ни было GPU. Число приходит снаружи: планировщик
намеренно ничего не знает ни про /proc, ни про железо. ``None`` — без потолка.

Задача — это dict как минимум с "type", "ts" (unix-время постановки) и "user"
(id владельца; None — общий анонимный пользователь, который в круге и под
потолком участвует наравне с остальными).
"""
import threading
import time


# None как «узел не задан» не годится: это законный ключ локального узла.
_UNSET = object()


def user_of(job):
    """Владелец задачи.

    Задачи без владельца делят одного общего анонимного пользователя: и место
    в круге, и потолок у них общие, как у любого другого человека.
    """
    return job.get("user")


class Device:
    """Состояние одной карты: что на ней тёплое и сколько подряд обслужено.

    Живёт на устройство, а не на планировщик: у каждой карты своя резидентная
    модель, и батчинг одной не должен влиять на выбор другой.
    """

    __slots__ = ("resident_vtype", "subtype_streak", "video_streak")

    def __init__(self):
        self.resident_vtype = None   # видео-подтип, сейчас загруженный в слот
        self.subtype_streak = 0      # видео этого подтипа обслужено подряд
        self.video_streak = 0        # видео любого типа обслужено подряд

    def copy(self):
        other = Device()
        for name in self.__slots__:
            setattr(other, name, getattr(self, name))

        return other


class Policy:
    """Состояние политики: круг по людям (общий) и карты (каждая со своим).

    Отдельным объектом, потому что нужно дважды: планировщику — для настоящего
    выбора, снимку — для прогона будущего порядка на копии.
    """

    __slots__ = ("user_order", "current_user", "user_streak", "devices")

    def __init__(self):
        self.user_order = []         # круг: порядок, в котором доходит очередь
        self.current_user = None     # кого обслуживаем прямо сейчас
        self.user_streak = 0         # сколько его задач обслужено подряд
        self.devices = {}            # ключ устройства -> Device

    def device(self, device=None):
        """Состояние карты; заводится при первом обращении.

        Ключ ``None`` — единственная карта. Так однокарточная конфигурация
        остаётся частным случаем пула, а не отдельной веткой кода.
        """
        state = self.devices.get(device)
        if state is None:
            state = self.devices[device] = Device()

        return state

    def copy(self):
        other = Policy()
        other.current_user = self.current_user
        other.user_streak = self.user_streak
        other.user_order = list(self.user_order)
        other.devices = {k: v.copy() for k, v in self.devices.items()}

        return other

    def record(self, job, video_types, device=None):
        """Учитывает обслуженную задачу на карте ``device``."""
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

        dev = self.device(device)
        jtype = job["type"]
        if jtype in video_types:
            dev.video_streak += 1
            if jtype == dev.resident_vtype:
                dev.subtype_streak += 1
            else:
                dev.resident_vtype = jtype
                dev.subtype_streak = 1
        else:
            # лёгкая задача сбрасывает счётчик видео подряд; видео-слот не
            # трогаем, поэтому resident_vtype/subtype_streak сохраняются
            dev.video_streak = 0


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


def startable(pending, video_types, *, video_running=0,
              max_concurrent_video=None, allowed_types=None,
              video_allowed=None):
    """Задачи, которые можно начать прямо сейчас на этой карте.

    ``allowed_types`` — что карта вообще умеет. Карты пула перестали быть
    однородными: у удалённой есть только чужой ComfyUI, а diffusers-процесса
    там нет и быть не может, поэтому картинки и транскрипцию ей отдавать
    нельзя. None — умеет всё, как было до нод.

    ``video_allowed`` — готовое решение «можно ли начать видео», когда потолок
    считается ПО УЗЛУ. Оперативная память принадлежит машине: на одном хосте
    помещается два ролика, на другом один, и общего числа на пул больше не
    существует. Задан — перекрывает расчёт по ``video_running``.
    """
    ready = pending if allowed_types is None else [
        j for j in pending if j["type"] in allowed_types]

    if video_allowed is None:
        video_allowed = (max_concurrent_video is None
                         or video_running < max_concurrent_video)
    if video_allowed:
        return ready

    return [j for j in ready if j["type"] not in video_types]


def pick_job(pending, video_types, policy, *, max_video_batch, max_wait_secs,
             max_videos_before_cheap, max_user_batch, device=None,
             video_running=0, max_concurrent_video=None, now=None,
             allowed_types=None, video_allowed=None):
    """Чистая функция выбора следующей задачи для карты ``device``.

    Возвращает выбранный элемент ``pending`` (не удаляя его) либо ``None``,
    если начать сейчас нечего: ждут одни видео, а потолок одновременных видео
    уже выбран. Карте в этом случае остаётся ждать освобождения места, а не
    хвататься за задачу, которую всё равно не потянуть.
    """
    if now is None:
        now = time.time()

    ready = startable(pending, video_types, video_running=video_running,
                      max_concurrent_video=max_concurrent_video,
                      allowed_types=allowed_types, video_allowed=video_allowed)
    if not ready:
        return None

    dev = policy.device(device)
    user = pick_user(ready, policy, max_user_batch=max_user_batch)
    mine = [j for j in ready if user_of(j) == user]

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
    if cheap and dev.video_streak >= max_videos_before_cheap:
        return min(cheap, key=lambda j: j["ts"])

    # Держим видео-модель тёплой: добиваем задачи резидентного видео-подтипа
    # ЭТОЙ карты. Соседняя со своей моделью на выбор не влияет — она разберёт
    # то, что тёплое у неё, и специализация складывается сама собой.
    if dev.resident_vtype in video_types and dev.subtype_streak < max_video_batch:
        same = [j for j in mine if j["type"] == dev.resident_vtype]
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

    def __init__(self, video_types, *, devices=(None,), max_video_batch=10,
                 max_wait_secs=900, max_videos_before_cheap=3, max_user_batch=2,
                 max_user_inflight=5, max_concurrent_video=None,
                 device_nodes=None, node_video_caps=None, device_types=None):
        self.video_types = tuple(video_types)
        self.devices = tuple(devices)
        self.max_concurrent_video = max_concurrent_video
        # Карта -> узел, на котором она стоит. Пусто — все карты на одной
        # машине, и потолок остаётся общим на пул, как было до нод.
        self.device_nodes = dict(device_nodes or {})
        # Узел -> сколько видео он держит одновременно. Считается из его
        # оперативной памяти, а не из числа карт.
        self.node_video_caps = dict(node_video_caps or {})
        # Карта -> что она умеет. Отсутствует для карты — умеет всё.
        self.device_types = {k: frozenset(v)
                             for k, v in (device_types or {}).items()}
        self.max_video_batch = max_video_batch
        self.max_wait_secs = max_wait_secs
        self.max_videos_before_cheap = max_videos_before_cheap
        self.max_user_batch = max_user_batch
        self.max_user_inflight = max_user_inflight

        self._pending = []
        self._inflight = {}   # пользователь -> задач в работе (в очереди + на счёте)
        self._running = {}    # карта -> задача, которую она считает прямо сейчас
        self._cv = threading.Condition()
        self._stopping = False

        # состояние политики (публичное — удобно смотреть и проверять в тестах)
        self.policy = Policy()

    # ------------------------------------------------------------------
    #  Снаружи и в тестах эти три читаются как поля планировщика. При
    #  нескольких картах они относятся к первой из ``devices`` — для пула
    #  смотри ``snapshot()["devices"]``.
    # ------------------------------------------------------------------
    @property
    def resident_vtype(self):
        return self.policy.device(self.devices[0]).resident_vtype

    @property
    def subtype_streak(self):
        return self.policy.device(self.devices[0]).subtype_streak

    @property
    def video_streak(self):
        return self.policy.device(self.devices[0]).video_streak

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
        """Задача досчитана: освобождает место в допуске и карту под следующую.

        Будим всех: место могло освободиться не только под ту карту, что
        закончила, — если упирались в потолок одновременных видео, ждать могли
        и остальные.
        """
        user = user_of(job)

        with self._cv:
            for device, running in list(self._running.items()):
                if running is job:
                    del self._running[device]
                    break

            left = self._inflight.get(user, 0) - 1
            if left > 0:
                self._inflight[user] = left
            else:
                self._inflight.pop(user, None)
                self._prune_order()

            self._cv.notify_all()

    def _video_running(self, exclude=None, node=_UNSET):
        """Сколько видео считается прямо сейчас (вызывать под ``self._cv``).

        Карту, которая как раз спрашивает себе работу, из счёта исключаем: она
        свободна, что бы там ни осталось в ``_running`` от прошлой задачи.

        ``node`` задан — считаем только на этом узле: память принадлежит
        машине, и ролики на соседнем хосте нашему ничем не мешают.
        """
        return sum(1 for device, job in self._running.items()
                   if device != exclude and job["type"] in self.video_types
                   and (node is _UNSET
                        or self.device_nodes.get(device) == node))

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

    def _pick(self, now=None, device=None):
        """Выбор без удаления и без учёта (вызывать под ``self._cv``)."""
        return pick_job(
            self._pending, self.video_types, self.policy,
            max_video_batch=self.max_video_batch,
            max_wait_secs=self.max_wait_secs,
            max_videos_before_cheap=self.max_videos_before_cheap,
            max_user_batch=self.max_user_batch,
            device=device,
            video_running=self._video_running(exclude=device),
            max_concurrent_video=self.max_concurrent_video,
            now=now,
            allowed_types=self.device_types.get(device),
            video_allowed=self._video_allowed(device),
        )

    def _video_allowed(self, device):
        """Можно ли этой карте начать видео (вызывать под ``self._cv``).

        Два условия, и держаться должны оба: потолок узла (сколько роликов
        влезает в память ЭТОЙ машины) и потолок пула, если он задан. Без
        описания узлов возвращает None — тогда решает потолок пула, ровно как
        было раньше.
        """
        node = self.device_nodes.get(device)
        cap = self.node_video_caps.get(node)
        if cap is None:
            return None

        if self._video_running(exclude=device, node=node) >= cap:
            return False

        pool = self.max_concurrent_video
        return pool is None or self._video_running(exclude=device) < pool

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
        step = 0
        while pending:
            # карты разбирают очередь по кругу; потолок видео в прогоне не
            # учитываем — он сдвигает СТАРТ во времени, а не место в очереди,
            # а снаружи спрашивают именно «какой я по счёту»
            device = self.devices[step % len(self.devices)]
            job = pick_job(
                pending, self.video_types, policy,
                max_video_batch=self.max_video_batch,
                max_wait_secs=self.max_wait_secs,
                max_videos_before_cheap=self.max_videos_before_cheap,
                max_user_batch=self.max_user_batch,
                device=device,
                now=now,
            )
            if job is None:
                break

            pending.remove(job)
            policy.record(job, self.video_types, device)
            order.append(job)
            step += 1

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
                "devices": {
                    device: {
                        "resident_vtype": self.policy.device(device).resident_vtype,
                        "subtype_streak": self.policy.device(device).subtype_streak,
                        "video_streak": self.policy.device(device).video_streak,
                        "running": (self._running.get(device) or {}).get("id"),
                    }
                    for device in self.devices
                },
                "video_running": self._video_running(),
                "max_concurrent_video": self.max_concurrent_video,
                "nodes": {
                    node: {
                        "video_cap": cap,
                        "video_running": self._video_running(node=node),
                        "devices": [d for d in self.devices
                                    if self.device_nodes.get(d) == node],
                    }
                    for node, cap in self.node_video_caps.items()
                },
                # три поля ниже — про первую карту; оставлены, чтобы не ломать
                # тех, кто читал снимок до появления пула
                "resident_vtype": self.resident_vtype,
                "subtype_streak": self.subtype_streak,
                "video_streak": self.video_streak,
                "current_user": self.policy.current_user,
                "user_streak": self.policy.user_streak,
                "inflight": dict(self._inflight),
                "next_id": order[0].get("id") if order else None,
            }

    def _take(self, now=None, device=None):
        """Небл. ядро: выбирает, удаляет и учитывает задачу для карты.

        ``None`` — брать нечего: либо очередь пуста, либо ждут одни видео при
        выбранном потолке. Общая основа для :meth:`next_job` и для тестов (там
        передают ``now``). Вызывающий обеспечивает отсутствие гонок (держит
        ``self._cv`` либо работает однопоточно).
        """
        if not self._pending:
            return None

        job = self._pick(now=now, device=device)
        if job is None:
            return None

        self._pending.remove(job)
        self.policy.record(job, self.video_types, device)
        self._running[device] = job

        return job

    def next_job(self, device=None):
        """Блокирующе ждёт задачу для карты ``device``; None при остановке.

        Ждём не только пустую очередь, но и занятый потолок видео: задача может
        лежать в очереди и всё равно быть не начинаемой прямо сейчас. Разбудит
        :meth:`enqueue` или :meth:`finish`.
        """
        with self._cv:
            while True:
                job = self._take(device=device)
                if job is not None:
                    return job
                if self._stopping:
                    return None

                self._cv.wait()
