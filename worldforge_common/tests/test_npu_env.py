import importlib.metadata

import pytest

from worldforge_common import npu_env
from worldforge_common.exceptions import VersionMismatchError


def test_check_versions_non_strict_returns_list(monkeypatch):
    """check_versions returns the exact (pkg, expected, actual) tuples for mismatches
    and an empty list when every locked package matches the mocked environment."""
    # Force every entry to mismatch with a known, controlled "actual" version.
    fake_installed = {
        "torch": "2.9.0",  # matches
        "torch_npu": "1.0.0",  # mismatch
        "transformers": "4.55.0",  # matches
        "peft": "0.0.1",  # mismatch
        "accelerate": "1.10.0",  # matches
    }

    def fake_version(pkg: str) -> str:
        return fake_installed[pkg]

    monkeypatch.setattr(npu_env.importlib.metadata, "version", fake_version)

    result = npu_env.check_versions(strict=False)
    assert result == [
        ("torch_npu", "2.9.0", "1.0.0"),
        ("peft", "0.18.0", "0.0.1"),
    ]


def test_check_versions_returns_empty_when_all_match(monkeypatch):
    """When every mocked version matches the locked matrix, check_versions
    must return an empty list — not just 'a list'."""
    def fake_version(pkg: str) -> str:
        return npu_env.LOCKED_VERSIONS[pkg]

    monkeypatch.setattr(npu_env.importlib.metadata, "version", fake_version)

    assert npu_env.check_versions(strict=False) == []


def test_check_versions_strict_raises_on_mismatch(monkeypatch):
    # Force a mismatch by tampering with LOCKED_VERSIONS
    monkeypatch.setitem(npu_env.LOCKED_VERSIONS, "torch", "999.999.999")
    with pytest.raises(VersionMismatchError):
        npu_env.check_versions(strict=True)


def test_check_versions_strict_absent_package_message(monkeypatch):
    """When a locked package is not installed, the strict-mode error must
    use MISSING_VERSION as the sentinel and explicitly say the package is
    not installed (rather than presenting the sentinel as a real version)."""
    monkeypatch.setitem(npu_env.LOCKED_VERSIONS, "torch_npu", "9.9.9")

    def fake_version(pkg: str) -> str:
        if pkg == "torch_npu":
            raise importlib.metadata.PackageNotFoundError(pkg)
        return npu_env.LOCKED_VERSIONS[pkg]

    monkeypatch.setattr(npu_env.importlib.metadata, "version", fake_version)

    with pytest.raises(VersionMismatchError) as excinfo:
        npu_env.check_versions(strict=True)

    msg = str(excinfo.value)
    # The user-facing message distinguishes "not installed" from a real mismatch
    # and names the offending package. (The sentinel itself is intentionally
    # not echoed — the message replaces it with clearer prose.)
    assert "not installed" in msg
    assert "torch_npu" in msg


def test_check_versions_absent_constant_used(monkeypatch):
    """The MISSING_VERSION constant must be what the absent-package path
    returns from _get_version — and it must appear in the mismatch tuple."""
    def fake_version(pkg: str) -> str:
        raise importlib.metadata.PackageNotFoundError(pkg)

    monkeypatch.setattr(npu_env.importlib.metadata, "version", fake_version)

    result = npu_env.check_versions(strict=False)
    # Every entry should use the constant as the actual version.
    assert all(actual == npu_env.MISSING_VERSION for _, _, actual in result)
    # And every locked package should be reported.
    reported = {pkg for pkg, _, _ in result}
    assert reported == set(npu_env.LOCKED_VERSIONS.keys())


def test_is_npu_available_returns_bool():
    result = npu_env.is_npu_available()
    assert isinstance(result, bool)


def test_is_npu_available_true_when_torch_npu_importable(monkeypatch):
    """When torch_npu imports and torch.npu.is_available() returns True,
    is_npu_available must report True — regardless of the real machine."""
    import sys
    import types

    fake_torch = types.ModuleType("torch")
    fake_npu = types.ModuleType("torch.npu")
    fake_npu.is_available = lambda: True  # type: ignore[attr-defined]
    fake_npu.device_count = lambda: 8  # type: ignore[attr-defined]
    fake_torch.npu = fake_npu  # type: ignore[attr-defined]

    fake_torch_npu = types.ModuleType("torch_npu")

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torch.npu", fake_npu)
    monkeypatch.setitem(sys.modules, "torch_npu", fake_torch_npu)

    assert npu_env.is_npu_available() is True


def test_is_npu_available_false_when_torch_npu_absent(monkeypatch):
    """When torch_npu cannot be imported, is_npu_available must report False."""
    import builtins
    import sys

    # Make sure torch_npu is not importable.
    monkeypatch.delitem(sys.modules, "torch_npu", raising=False)

    real_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "torch_npu" or name.startswith("torch_npu."):
            raise ImportError(f"No module named {name!r}")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    assert npu_env.is_npu_available() is False


def test_is_npu_available_propagates_torch_importerror(monkeypatch):
    """A genuine ImportError from torch itself must propagate — G8 forbids
    silently reporting 'no NPU' when the install is broken."""
    import builtins
    import sys

    real_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "torch":
            raise ImportError(f"No module named {name!r}")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setitem(sys.modules, "torch_npu", object())  # so torch_npu import succeeds
    monkeypatch.setattr(builtins, "__import__", guarded_import)

    with pytest.raises(ImportError):
        npu_env.is_npu_available()


def test_npu_device_count_returns_int():
    count = npu_env.npu_device_count()
    assert isinstance(count, int)
    assert count >= 0


def test_npu_device_count_returns_mocked_count_when_npu_available(monkeypatch):
    """When NPU is mocked-available, npu_device_count must return the mocked count."""
    import sys
    import types

    fake_torch = types.ModuleType("torch")
    fake_npu = types.ModuleType("torch.npu")
    fake_npu.is_available = lambda: True  # type: ignore[attr-defined]
    fake_npu.device_count = lambda: 8  # type: ignore[attr-defined]
    fake_torch.npu = fake_npu  # type: ignore[attr-defined]

    fake_torch_npu = types.ModuleType("torch_npu")

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torch.npu", fake_npu)
    monkeypatch.setitem(sys.modules, "torch_npu", fake_torch_npu)

    assert npu_env.npu_device_count() == 8


def test_npu_device_count_returns_zero_when_npu_absent(monkeypatch):
    """When NPU is not available, npu_device_count must return 0."""
    import builtins
    import sys

    monkeypatch.delitem(sys.modules, "torch_npu", raising=False)

    real_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "torch_npu" or name.startswith("torch_npu."):
            raise ImportError(f"No module named {name!r}")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    assert npu_env.npu_device_count() == 0