"""Обёртка над долгоживущим GPU-процессом.

Вынесена из main.py и НЕ тянет тяжёлых зависимостей (torch и пр.): целевая
функция процесса передаётся аргументом, поэтому механику submit/respawn можно
покрыть тестами с фейковым воркером, без видеокарты.

Контракт целевой функции ``target(job_q, res_q)``:
  * читает задачи из ``job_q`` (dict с ключом "id"); "BREAK" — сигнал выхода;
  * на каждую задачу кладёт в ``res_q`` кортеж ``(id, result)``.
Строго одна задача «в полёте» за раз (потребитель ждёт результат синхронно).
"""
import multiprocessing
import queue as _queue


class GpuRunner:
    """Держит долгоживущий процесс; поднимает заново, если тот умер."""

    def __init__(self, target, *, mp_context="spawn", poll_timeout=2):
        self.ctx = multiprocessing.get_context(mp_context)
        self.target = target
        self.poll_timeout = poll_timeout
        self.proc = None
        self.job_q = None
        self.res_q = None
        self._start()

    def _start(self):
        # свежие очереди, чтобы после смерти процесса не осталось «висящих» задач
        self.job_q = self.ctx.Queue()
        self.res_q = self.ctx.Queue()
        self.proc = self.ctx.Process(
            target=self.target, args=(self.job_q, self.res_q), daemon=True)
        self.proc.start()

    def submit_and_wait(self, job):
        """Отправляет задачу и блокирующе ждёт результат (строго по одной за раз)."""
        self.job_q.put(job)
        while True:
            try:
                _id, res = self.res_q.get(timeout=self.poll_timeout)
                return res
            except _queue.Empty:
                if not self.proc.is_alive():
                    code = self.proc.exitcode
                    print(f"[gpu] process died (exitcode={code}), respawning", flush=True)
                    self._start()  # поднимаем заново для следующих задач
                    return {"error": f"generation process died (exitcode={code})"}

    def recycle(self):
        """Жёсткий сброс VRAM: убить процесс (CUDA-контекст уничтожается вместе с
        ним) и поднять заново. Единственный надёжный способ полностью вернуть VRAM
        — внутри живого процесса CUDA оставляет «хвост», который empty_cache() не
        отдаёт. Вызывается при смене модели."""
        self._kill()
        self._start()

    def _kill(self, timeout=10):
        if self.proc is None:
            return
        if self.proc.is_alive():
            self.proc.terminate()               # SIGTERM
            self.proc.join(timeout=timeout)
            if self.proc.is_alive():
                self.proc.kill()                # SIGKILL, если не завершился
                self.proc.join(timeout=timeout)

    def stop(self):
        try:
            self.job_q.put("BREAK")
        except Exception:
            pass
