from pyrecest.evaluation.get_distance_function import get_distance_function


def test_hypersphere_symmetric_accepts_list_values():
    distance = get_distance_function("hypersphereSymmetric")

    assert float(distance([1.0, 0.0], [-1.0, 0.0])) == 0.0


def test_se3bounded_accepts_list_quaternions():
    distance = get_distance_function("se3bounded")
    estimate = [1.0, 0.0, 0.0, 0.0, 10.0, 20.0, 30.0]
    expected = [-1.0, 0.0, 0.0, 0.0, -5.0, 3.0, 2.0]

    assert float(distance(estimate, expected)) == 0.0
