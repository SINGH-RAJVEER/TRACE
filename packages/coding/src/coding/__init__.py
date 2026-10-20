"""Optional coding extension. Importing this package never executes candidate code."""

import os

DEFAULT_QUALIFICATION_WORKERS = min(16, os.process_cpu_count() or 1)
DEFAULT_POOL_SIZE = 50_000
