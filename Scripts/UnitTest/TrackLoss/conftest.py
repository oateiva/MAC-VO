import pytest

from asset_trackloss import build_sequence, seeded_index


@pytest.fixture(scope="session")
def sequence():
    """The real plane_nose monocular sequence. Frame loading is lazy, so this is cheap."""
    return build_sequence()


@pytest.fixture(scope="session")
def anchor_idx(sequence) -> int:
    return seeded_index(len(sequence))
