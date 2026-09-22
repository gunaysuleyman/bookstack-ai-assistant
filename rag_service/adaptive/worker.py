import logging
import threading
from typing import Callable, Optional

from adaptive.config import Settings
from adaptive.store import StateStore

logger = logging.getLogger("IndexWorker")


class IndexWorker:
    def __init__(self, store: StateStore, settings: Settings, handler: Callable):
        self.store = store
        self.settings = settings
        self.handler = handler
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def tick(self) -> bool:
        try:
            self.store.purge_expired_scopes()
        except Exception:
            logger.exception("Expired scope purge failed")
        job = self.store.lease_next("api-worker", self.settings.job_lease_seconds)
        if job is None:
            return False
        try:
            self.handler(job)
        except Exception as exc:
            logger.warning("Job %s failed: %s", job.id, exc)
            self.store.fail_job(job.id, str(exc))
            return True
        self.store.complete_job(job.id)
        return True

    def start(self, interval_s: float = 1.0) -> None:
        if self._thread and self._thread.is_alive():
            return

        def loop():
            while not self._stop.is_set():
                try:
                    worked = self.tick()
                except Exception:
                    logger.exception("Index worker tick failed")
                    worked = False
                self._stop.wait(0.05 if worked else interval_s)

        self._thread = threading.Thread(target=loop, name="index-worker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
