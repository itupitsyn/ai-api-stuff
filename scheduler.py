"""Планировщик очереди генерации.

Вынесен из main.py и НЕ тянет тяжёлых зависимостей (torch, diffusers, whisperx),
чтобы логику планирования можно было покрыть юнит-тестами где угодно.

Политика: по умолчанию FIFO, но пока видео-модель тёплая — добиваем задачи того
же видео-подтипа (чтобы не перегружать ~50 ГБ). Предохранители от голодания:
  * MAX_VIDEO_BATCH        — макс. видео одного подтипа подряд, если ждёт другой тип;
  * MAX_VIDEOS_BEFORE_CHEAP — макс. видео подряд, если ждут картинки/транскрипции;
  * MAX_WAIT_SECS          — задачу, ждущую дольше, обслуживаем вне батчинга.

Задача — это dict с ключами как минимум "type" и "ts" (unix-время постановки).
Картинки/транскрипция ("лёгкие") видео-слот не вытесняют, поэтому их можно
вклинивать между видео без потери тёплой модели.
"""
import threading
import time


def pick_job(pending, video_types, resident_vtype, subtype_streak, video_streak,
             *, max_video_batch, max_wait_secs, max_videos_before_cheap, now=None):
    """Чистая функция выбора следующей задачи из непустого списка ``pending``.

    Возвращает выбранный элемент ``pending`` (не удаляя его).
    """
    if now is None:
        now = time.time()

    oldest = min(pending, key=lambda j: j["ts"])

    # анти-старвейшн: слишком долго ждущую задачу обслуживаем немедленно
    if now - oldest["ts"] >= max_wait_secs:
        return oldest

    # не мариновать лёгкие задачи: после N видео подряд пропускаем вперёд
    # ожидающую картинку/транскрипцию (видео-слот при этом остаётся тёплым)
    cheap = [j for j in pending if j["type"] not in video_types]
    if cheap and video_streak >= max_videos_before_cheap:
        return min(cheap, key=lambda j: j["ts"])

    # держим видео-модель тёплой: добиваем задачи резидентного видео-подтипа
    if resident_vtype in video_types and subtype_streak < max_video_batch:
        same = [j for j in pending if j["type"] == resident_vtype]
        if same:
            return min(same, key=lambda j: j["ts"])

    return oldest


class Scheduler:
    """Потокобезопасная очередь с батчингом по типу.

    Продюсеры зовут :meth:`enqueue`; единственный потребитель крутит
    :meth:`next_job` (блокирующе). Счётчики батчинга обновляются в момент выбора
    задачи — потребитель один, поэтому гонок с последующей обработкой нет.
    """

    def __init__(self, video_types, *, max_video_batch=10, max_wait_secs=900,
                 max_videos_before_cheap=3):
        self.video_types = tuple(video_types)
        self.max_video_batch = max_video_batch
        self.max_wait_secs = max_wait_secs
        self.max_videos_before_cheap = max_videos_before_cheap

        self._pending = []
        self._cv = threading.Condition()
        self._stopping = False

        # состояние батчинга (публичное — удобно смотреть/проверять в тестах)
        self.resident_vtype = None   # видео-подтип, сейчас загруженный в слот
        self.subtype_streak = 0      # видео этого подтипа обслужено подряд
        self.video_streak = 0        # видео любого типа обслужено подряд

    def enqueue(self, job):
        job.setdefault("ts", time.time())
        with self._cv:
            self._pending.append(job)
            self._cv.notify()

    def stop(self):
        with self._cv:
            self._stopping = True
            self._cv.notify_all()

    def pending_count(self):
        with self._cv:
            return len(self._pending)

    def _record(self, jtype):
        """Обновляет счётчики после выбора задачи типа ``jtype``."""
        if jtype in self.video_types:
            self.video_streak += 1
            if jtype == self.resident_vtype:
                self.subtype_streak += 1
            else:
                self.resident_vtype = jtype
                self.subtype_streak = 1
        else:
            # лёгкая задача сбрасывает счётчик видео подряд; видео-слот не трогаем,
            # поэтому resident_vtype/subtype_streak сохраняются (модель тёплая)
            self.video_streak = 0

    def _take(self, now=None):
        """Небл. ядро: выбирает, удаляет и учитывает задачу. None, если пусто.

        Общая основа для :meth:`next_job` и для тестов (там передают ``now``).
        Вызывающий обеспечивает отсутствие гонок (держит ``self._cv`` либо
        работает однопоточно).
        """
        if not self._pending:
            return None
        job = pick_job(
            self._pending, self.video_types, self.resident_vtype,
            self.subtype_streak, self.video_streak,
            max_video_batch=self.max_video_batch,
            max_wait_secs=self.max_wait_secs,
            max_videos_before_cheap=self.max_videos_before_cheap,
            now=now,
        )
        self._pending.remove(job)
        self._record(job["type"])
        return job

    def next_job(self):
        """Блокирующе ждёт и возвращает следующую задачу; None при остановке."""
        with self._cv:
            while not self._pending and not self._stopping:
                self._cv.wait()
            if self._stopping and not self._pending:
                return None
            return self._take()
