import time
import collections
from typing import Dict, List
import numpy as np


class FpsCounterJob:
    def __init__(self, average_of=30):
        self._average_of = average_of
        self._frame_durations = collections.deque(maxlen=average_of)
        self._start_time: float = None

    def start(self):
        self._start_time = time.time()

    def end(self):
        self._frame_durations.append(time.time() - self._start_time)

    def mean_frame_time(self):
        return np.mean(self._frame_durations)

    def measure(self):
        mean_frame_time = self.mean_frame_time()

        if mean_frame_time > 0:
            return 1.0 / self.mean_frame_time()

        return 0


class FpsCounter:
    def __init__(self, average_of=30):
        self._average_of = average_of
        self._frame = FpsCounterJob(self._average_of)
        self.jobs: Dict[str, FpsCounterJob] = {}

    def start_job(self, job: str):
        if job not in self.jobs:
            self.jobs[job] = FpsCounterJob(self._average_of)

        self.jobs[job].start()

    def end_job(self, job: str):
        self.jobs[job].end()

    def measure(self, exclude_jobs: List[str] = []):
        mean_frame_time = self.mean_frame_time(exclude_jobs)

        if mean_frame_time > 0:
            return 1.0 / mean_frame_time

        return 0

    def mean_frame_time(self, exclude_jobs: List[str] = []):
        frame_time = self._frame.mean_frame_time()

        for job in self.jobs:
            if job in exclude_jobs:
                frame_time -= self.jobs[job].mean_frame_time()

        return frame_time

    def start_frame(self):
        self._frame.start()

    def end_frame(self):
        self._frame.end()

    def __str__(self):
        status = f"{round(self.measure(), 1)}fps"

        for name, job in self.jobs.items():
            status += f", {name}={round(job.mean_frame_time()*1000)}ms"
        
        return status
