"""Fixed local subprocess entry point; invoked only by the trusted refresh controller."""

import sys
from pathlib import Path

from oddsjam_ev.dashboard.refresh import execute

if __name__ == "__main__":
    execute(Path(sys.argv[1]))
