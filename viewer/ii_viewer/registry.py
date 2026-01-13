import threading


LIVE_RUNS_LOCK = threading.Lock()
LIVE_RUNS = {}

MASK_JOBS_LOCK = threading.Lock()
MASK_JOBS = {}
