import numpy as np
import pytest
from pyrecest._backend.numpy import random


def test_rand_accepts_backend_size_keyword():
    random.seed(0)

    samples = random.rand(size=(2, 3))

    assert samples.shape == (2, 3)
    assert samples.dtype == np.float64


def test_rand_keeps_numpy_positional_dimensions():
    random.seed(0)

    assert random.rand(2, 3).shape == (2, 3)
    assert random.rand(4).shape == (4,)


def test_rand_rejects_ambiguous_positional_and_size_arguments():
    with pytest.raises(TypeError, match="positional dimensions or size"):
        random.rand(2, size=(3,))


@pytest.mark.parametrize(
    ("low", "high"),
    [
        (False, 1.0),
        (0.0, True),
        (np.array([False, False]), np.array([1.0, 2.0])),
    ],
)
def test_uniform_rejects_boolean_bounds(low, high):
    with pytest.raises(TypeError, match="real numeric"):
        random.uniform(low, high)


@pytest.mark.parametrize(
    ("low", "high"),
    [
        ("0.0", 1.0),
        (0.0, "1.0"),
        (np.array(["0.0", "0.5"]), np.array([1.0, 1.5])),
    ],
)
def test_uniform_rejects_text_bounds(low, high):
    with pytest.raises(TypeError, match="real numeric"):
        random.uniform(low, high)


def test_choice_without_replacement_shuffle_false_preserves_order():
    values = np.array([10, 20, 30, 40, 50])
    matrix = np.array([[10, 20, 30], [40, 50, 60]])

    random.seed(0)
    samples = random.choice(values, size=values.shape[0], replace=False, shuffle=False)
    column_samples = random.choice(
        matrix,
        size=matrix.shape[1],
        replace=False,
        axis=1,
        shuffle=False,
    )

    np.testing.assert_array_equal(samples, values)
    np.testing.assert_array_equal(column_samples, matrix)


def test_choice_without_replacement_shuffle_false_preserves_population_order_for_unsorted_values():
    values = np.array([30, 10, 20, 40])

    random.seed(1)
    expected_indices = np.sort(np.random.choice(values.shape[0], size=3, replace=False))
    expected = values[expected_indices]

    random.seed(1)
    samples = random.choice(values, size=3, replace=False, shuffle=False)

    np.testing.assert_array_equal(samples, expected)


@pytest.mark.parametrize("control", ["replace", "shuffle"])
def test_choice_rejects_non_boolean_controls(control):
    kwargs = {control: "False"}

    with pytest.raises(TypeError, match=f"{control} must be a boolean"):
        random.choice(np.arange(3), size=2, **kwargs)
