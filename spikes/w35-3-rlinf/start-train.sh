#!/bin/bash
# W35.3 spike baseline launcher. Start with:
#   setsid bash start-train.sh < /dev/null > /dev/null 2>&1 &
D=/mnt/c/WorkSpace/WorldForge/spikes/w35-3-rlinf
V=/home/chad/w35-3/venv
cd "$D" || exit 1
export RAY_ADDRESS=127.0.0.1:6399
export PYTHONPATH="$D"
exec "$V/bin/python" src/train_spike.py > "$D/run-baseline.log" 2>&1
