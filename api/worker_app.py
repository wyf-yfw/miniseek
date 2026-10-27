"""MiniSeek worker control plane — owns train/bench Popen lifecycle."""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from api.task_runner import (
    reattach_running_tails,
    start_task,
    stop_all_owned as stop_all_train,
    stop_task,
)
from api.bench_runner import (
    reattach_running as reattach_bench,
    start_run as bench_start_run,
    stop_all_owned as stop_all_bench,
    stop_run as bench_stop_run,
)


class BenchStartBody(BaseModel):
    training_busy: bool = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    reattach_running_tails()
    reattach_bench()
    try:
        yield
    finally:
        stop_all_train()
        stop_all_bench()


app = FastAPI(title="MiniSeek Worker", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.post("/v1/tasks/{task_id}/start")
def worker_start_task(task_id: str) -> dict:
    try:
        return start_task(task_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (ValueError, FileNotFoundError, NotImplementedError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/v1/tasks/{task_id}/stop")
def worker_stop_task(task_id: str) -> dict:
    try:
        return stop_task(task_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/v1/bench/{run_id}/start")
def worker_start_bench(run_id: str, body: Optional[BenchStartBody] = None) -> dict:
    busy = bool(body.training_busy) if body is not None else False
    try:
        return bench_start_run(run_id, training_busy=busy)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (ValueError, NotImplementedError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/v1/bench/{run_id}/stop")
def worker_stop_bench(run_id: str) -> dict:
    try:
        return bench_stop_run(run_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
