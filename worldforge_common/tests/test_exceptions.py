from worldforge_common.exceptions import (
    AdapterNotFoundError,
    MaskingError,
    ModelNotFoundError,
    OMCompileError,
    VersionMismatchError,
    WorldForgeError,
)


def test_all_inherit_from_base():
    for cls in (
        VersionMismatchError,
        ModelNotFoundError,
        OMCompileError,
        AdapterNotFoundError,
        MaskingError,
    ):
        assert issubclass(cls, WorldForgeError)