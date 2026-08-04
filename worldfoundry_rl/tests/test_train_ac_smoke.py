import subprocess
import sys


def test_train_ac_help_runs():
    result = subprocess.run(
        [sys.executable, "-m", "worldforge_rl.train_ac", "--help"],
        capture_output=True, text=True, cwd="/mnt/c/WorkSpace/WorldForge",
    )
    assert result.returncode == 0
    assert "--model" in result.stdout