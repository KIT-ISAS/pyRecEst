def _unsupported_rotation_error(operation: str) -> RuntimeError:
    return RuntimeError(
        f"{operation} is not supported on the PyTorch backend. "
        "Use a NumPy/JAX backend for this functionality."
    )


class Rotation:
    """Dummy Rotation for PyTorch backend."""

    def __init__(self, *args, **kwargs):
        raise _unsupported_rotation_error("Rotation")

    @classmethod
    def from_quat(cls, *args, **kwargs):
        raise _unsupported_rotation_error("Rotation.from_quat")

    @classmethod
    def from_matrix(cls, *args, **kwargs):
        raise _unsupported_rotation_error("Rotation.from_matrix")

    @classmethod
    def from_rotvec(cls, *args, **kwargs):
        raise _unsupported_rotation_error("Rotation.from_rotvec")

    @classmethod
    def from_euler(cls, *args, **kwargs):
        raise _unsupported_rotation_error("Rotation.from_euler")

    @classmethod
    def identity(cls, *args, **kwargs):
        raise _unsupported_rotation_error("Rotation.identity")

    @classmethod
    def random(cls, *args, **kwargs):
        raise _unsupported_rotation_error("Rotation.random")

    @classmethod
    def align_vectors(cls, *args, **kwargs):
        raise _unsupported_rotation_error("Rotation.align_vectors")

    @classmethod
    def concatenate(cls, *args, **kwargs):
        raise _unsupported_rotation_error("Rotation.concatenate")

    def as_quat(self, *args, **kwargs):
        raise _unsupported_rotation_error("Rotation.as_quat")

    def as_matrix(self, *args, **kwargs):
        raise _unsupported_rotation_error("Rotation.as_matrix")

    def as_rotvec(self, *args, **kwargs):
        raise _unsupported_rotation_error("Rotation.as_rotvec")

    def as_euler(self, *args, **kwargs):
        raise _unsupported_rotation_error("Rotation.as_euler")

    def apply(self, *args, **kwargs):
        raise _unsupported_rotation_error("Rotation.apply")

    def inv(self, *args, **kwargs):
        raise _unsupported_rotation_error("Rotation.inv")

    def magnitude(self, *args, **kwargs):
        raise _unsupported_rotation_error("Rotation.magnitude")

    def mean(self, *args, **kwargs):
        raise _unsupported_rotation_error("Rotation.mean")

    def __mul__(self, other):
        raise _unsupported_rotation_error("Rotation composition")
