import subprocess
import sys


def test_train_help_runs():
    """The script must at least respond to --help without import errors."""
    result = subprocess.run(
        [sys.executable, "-m", "worldfoundry_train.train", "--help"],
        capture_output=True, text=True, cwd="/mnt/c/WorkSpace/WorldForge",
    )
    # Acceptable: exits 0 with help text
    assert result.returncode == 0, result.stderr
    assert "--parallel" in result.stdout
