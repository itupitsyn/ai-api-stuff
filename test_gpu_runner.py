"""Smoke-тесты механики GpuRunner (без GPU/torch).

Проверяют submit/return и respawn после смерти процесса с фейковыми воркерами.
Воркеры — top-level функции (иначе spawn не сможет их импортировать в потомке).

Запуск:  pytest test_gpu_runner.py -v
"""
import os

from gpu_runner import GpuRunner


# --- фейковые воркеры (контракт: читать job из job_q, класть (id, res) в res_q) ---

def echo_worker(job_q, res_q):
    while True:
        job = job_q.get()
        if job == "BREAK":
            break
        res_q.put((job["id"], {"echo": job["data"]}))


def dying_worker(job_q, res_q):
    while True:
        job = job_q.get()
        if job == "BREAK":
            break
        if job["data"] == "die":
            os._exit(1)  # жёсткая смерть без ответа (эмуляция OOM/segfault)
        res_q.put((job["id"], {"ok": job["data"]}))


def _shutdown(runner):
    runner.stop()
    if runner.proc is not None:
        runner.proc.join(timeout=5)


def test_happy_path_returns_result():
    runner = GpuRunner(echo_worker, poll_timeout=0.5)
    try:
        res = runner.submit_and_wait({"id": "1", "data": "hello"})
        assert res == {"echo": "hello"}
        # процесс переживает запрос и обслуживает следующий
        res2 = runner.submit_and_wait({"id": "2", "data": "world"})
        assert res2 == {"echo": "world"}
    finally:
        _shutdown(runner)


def test_respawn_after_death():
    runner = GpuRunner(dying_worker, poll_timeout=0.5)
    try:
        # процесс умирает, не ответив → получаем внятную ошибку, а не зависание
        res = runner.submit_and_wait({"id": "1", "data": "die"})
        assert isinstance(res, dict) and "died" in res["error"]

        # раннер поднял процесс заново — следующий запрос обслуживается
        res2 = runner.submit_and_wait({"id": "2", "data": "alive"})
        assert res2 == {"ok": "alive"}
    finally:
        _shutdown(runner)


def test_stop_terminates_process():
    runner = GpuRunner(echo_worker, poll_timeout=0.5)
    runner.stop()
    runner.proc.join(timeout=5)
    assert not runner.proc.is_alive()
