#!/bin/bash
cd /Users/avi/Projects/robot-sim
source .venv/bin/activate
export DYLD_LIBRARY_PATH=/opt/homebrew/Cellar/ffmpeg@7/7.1.3_2/lib:${DYLD_LIBRARY_PATH:-}
python scripts/train.py "$@"
