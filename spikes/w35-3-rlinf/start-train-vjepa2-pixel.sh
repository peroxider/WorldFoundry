#!/bin/bash
# W35.3 S2/S3 spike launcher (VJEPA2Policy mode A). Start with:
#   setsid bash start-train-vjepa2.sh < /dev/null > /dev/null 2>&1 &
D=/mnt/c/WorkSpace/WorldFoundry/spikes/w35-3-rlinf
V=/home/chad/w35-3/venv
cd "$D" || exit 1
export RAY_ADDRESS=127.0.0.1:6399
export PYTHONPATH="$D"
# Official extension channel (PR #920): every Worker process imports this
# module and calls register() to register VJEPA2Policy.
export RLINF_EXT_MODULE=spike_ext.rlinf_ext
exec "$V/bin/python" src/train_spike_vjepa2_pixel.py > "$D/run-vjepa2-pixel.log" 2>&1
