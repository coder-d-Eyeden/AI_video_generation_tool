"""
Lightweight background job runner for long video generation / render tasks.
"""
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional


class Job:
    def __init__(self, kind: str, meta: Optional[Dict[str, Any]] = None):
        self.id = uuid.uuid4().hex[:12]
        self.kind = kind
        self.status = "queued"
        self.progress = 0
        self.message = "Queued"
        self.result: Any = None
        self.error: Optional[str] = None
        self.meta = meta or {}
        self.created_at = time.time()
        self.updated_at = self.created_at

    def update(self, progress: Optional[int] = None, message: Optional[str] = None) -> None:
        if progress is not None:
            self.progress = max(0, min(100, int(progress)))
        if message is not None:
            self.message = message
        self.updated_at = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "status": self.status,
            "progress": self.progress,
            "message": self.message,
            "result": self.result,
            "error": self.error,
            "meta": self.meta,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class JobManager:
    def __init__(self, max_workers: int = 2):
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="video-job")
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()

    def submit(self, kind: str, fn: Callable[[Job], Any], meta: Optional[Dict[str, Any]] = None) -> Job:
        job = Job(kind, meta)
        with self._lock:
            self._jobs[job.id] = job
        self._executor.submit(self._run, job, fn)
        return job

    @staticmethod
    def _run(job: Job, fn: Callable[[Job], Any]) -> None:
        job.status = "running"
        job.update(message="Started")
        try:
            job.result = fn(job)
            job.status = "completed"
            job.update(100, job.message if job.message != "Started" else "Completed")
        except Exception as e:
            job.status = "failed"
            job.error = str(e) or e.__class__.__name__
            job.update(message="Failed")
            traceback.print_exc()

    def get(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    def list(self) -> List[Dict[str, Any]]:
        return [j.to_dict() for j in sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)]


job_manager = JobManager()
