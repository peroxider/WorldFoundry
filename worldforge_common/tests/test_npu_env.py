import pytest

from worldforge_common import npu_env
from worldforge_common.exceptions import VersionMismatchError


def test_check_versions_non_strict_returns_list():
    mismatches = npu_env.check_versions(strict=False)
    assert isinstance(mismatches, list)


def test_check_versions_strict_raises_on_mismatch(monkeypatch):
    # Force a mismatch by tampering with LOCKED_VERSIONS
    monkeypatch.setitem(npu_env.LOCKED_VERSIONS, "torch", "999.999.999")
    with pytest.raises(VersionMismatchError):
        npu_env.check_versions(strict=True)


def test_is_npu_available_returns_bool():
    result = npu_env.is_npu_available()
    assert isinstance(result, bool)


def test_npu_device_count_returns_int():
    count = npu_env.npu_device_count()
    assert isinstance(count, int)
    assert count >= 0
