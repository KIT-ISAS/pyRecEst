import numpy as np
import numpy.testing as npt
from pyrecest.backend import array, diag, eye
from pyrecest.filters.mem_qkf_tracker import MEMQKFTracker
from pyrecest.smoothers import (
    FBFBMEMQKFSmoother,
    FixedIntervalMEMQKFSmoother,
    FixedIntervalMemQkfSmoother,
    FixedLagFreeMEMQKFSmoother,
    FixedLagMEMQKFSmoother,
    FixedLagMemQkfSmoother,
    FLMEMQKFSmoother,
    ForwardBackwardForwardBackwardMEMQKFSmoother,
    ForwardBackwardMEMQKFSmoother,
    FullIntervalMEMQKFSmoother,
    MEMQKFTrackerState,
)


def _state(
    kinematic_state,
    covariance,
    shape_state,
    shape_covariance,
    *,
    minimum_axis_length=1e-9,
    minimum_covariance_eigenvalue=0.0,
):
    return MEMQKFTrackerState(
        array(kinematic_state),
        array(covariance),
        array(shape_state),
        array(shape_covariance),
        minimum_axis_length=minimum_axis_length,
        minimum_covariance_eigenvalue=minimum_covariance_eigenvalue,
    )


def test_aliases():
    assert FixedLagMemQkfSmoother is FixedLagMEMQKFSmoother
    assert FixedLagFreeMEMQKFSmoother is FixedLagMEMQKFSmoother
    assert FLMEMQKFSmoother is FixedLagMEMQKFSmoother
    assert FixedIntervalMemQkfSmoother is FixedIntervalMEMQKFSmoother
    assert FullIntervalMEMQKFSmoother is FixedIntervalMEMQKFSmoother
    assert FBFBMEMQKFSmoother is ForwardBackwardForwardBackwardMEMQKFSmoother
    assert ForwardBackwardMEMQKFSmoother is ForwardBackwardForwardBackwardMEMQKFSmoother


def test_lag_one_smooths_kinematics_and_mem_qkf_shape_state():
    smoother = FixedLagMEMQKFSmoother(lag=1)
    filtered_states = [
        _state(
            [0.0, 0.0, 1.0, 0.0],
            diag(array([0.5, 0.5, 0.5, 0.5])),
            [0.1, 2.0, 1.0],
            diag(array([0.2, 0.2, 0.2])),
        ),
        _state(
            [1.0, -1.0, 3.0, 0.0],
            diag(array([0.4, 0.4, 0.4, 0.4])),
            [0.5, 3.0, 2.0],
            diag(array([0.1, 0.1, 0.1])),
        ),
    ]
    predicted_states = [
        _state(
            [0.0, 0.0, 1.0, 0.0],
            diag(array([1.0, 1.0, 1.0, 1.0])),
            [0.1, 2.0, 1.0],
            diag(array([0.4, 0.4, 0.4])),
        )
    ]

    smoothed_states, smoother_gains = smoother.smooth(
        filtered_states=filtered_states,
        predicted_states=predicted_states,
        system_matrices=eye(4),
        shape_system_matrices=eye(3),
    )

    npt.assert_allclose(
        smoothed_states[0].kinematic_state,
        array([0.5, -0.5, 2.0, 0.0]),
    )
    npt.assert_allclose(
        smoothed_states[0].shape_state,
        array([0.3, 2.5, 1.5]),
        rtol=1e-6,
        atol=1e-7,
    )
    npt.assert_allclose(smoother_gains[0][0].kinematic, 0.5 * eye(4))
    npt.assert_allclose(smoother_gains[0][0].shape, 0.5 * eye(3))


def test_shape_smoothing_none_only_postprocesses_filtered_shape_state():
    smoother = FixedLagMEMQKFSmoother(lag=1, shape_smoothing="none")
    filtered_states = [
        _state(
            [0.0, 0.0, 1.0, 0.0],
            diag(array([0.5, 0.5, 0.5, 0.5])),
            [0.1, 2.0, 1.0],
            diag(array([0.2, 0.2, 0.2])),
        ),
        _state(
            [1.0, -1.0, 3.0, 0.0],
            diag(array([0.4, 0.4, 0.4, 0.4])),
            [0.5, 3.0, 2.0],
            diag(array([0.1, 0.1, 0.1])),
        ),
    ]
    predicted_states = [
        _state(
            [0.0, 0.0, 1.0, 0.0],
            diag(array([1.0, 1.0, 1.0, 1.0])),
            [0.1, 2.0, 1.0],
            diag(array([0.4, 0.4, 0.4])),
        )
    ]

    smoothed_states, smoother_gains = smoother.smooth(
        filtered_states,
        predicted_states,
        system_matrices=eye(4),
        shape_system_matrices=eye(3),
    )

    npt.assert_allclose(smoothed_states[0].shape_state, filtered_states[0].shape_state)
    assert smoother_gains[0][0].shape is None


def test_axial_orientation_residual_wraps_across_pi_period():
    smoother = FixedLagMEMQKFSmoother(lag=1)
    reference = np.pi - 0.05
    filtered_states = [
        _state(
            [0.0, 0.0, 0.0, 0.0],
            eye(4),
            [reference, 2.0, 1.0],
            diag(array([0.2, 0.2, 0.2])),
        ),
        _state(
            [0.0, 0.0, 0.0, 0.0],
            eye(4),
            [0.03, 2.0, 1.0],
            diag(array([0.1, 0.1, 0.1])),
        ),
    ]
    predicted_states = [
        _state(
            [0.0, 0.0, 0.0, 0.0],
            2.0 * eye(4),
            [reference, 2.0, 1.0],
            diag(array([0.4, 0.4, 0.4])),
        )
    ]

    smoothed_states, _ = smoother.smooth(
        filtered_states,
        predicted_states,
        system_matrices=eye(4),
        shape_system_matrices=eye(3),
    )

    expected_delta = ((0.03 - reference + np.pi / 2.0) % np.pi) - np.pi / 2.0
    npt.assert_allclose(
        smoothed_states[0].shape_state[0], reference + 0.5 * expected_delta
    )


def test_state_snapshot_round_trips_to_mem_qkf_tracker():
    tracker = MEMQKFTracker(
        kinematic_state=array([0.0, 0.0, 1.0, 0.0]),
        covariance=eye(4),
        shape_state=array([0.1, 2.0, 1.0]),
        shape_covariance=diag(array([0.2, 0.1, 0.1])),
        measurement_matrix=eye(2, 4),
        multiplicative_noise_cov=0.25 * eye(2),
        default_meas_noise_cov=0.05 * eye(2),
    )

    state = MEMQKFTrackerState.from_tracker(tracker)
    round_tripped = state.to_tracker()

    assert isinstance(round_tripped, MEMQKFTracker)
    npt.assert_allclose(round_tripped.kinematic_state, tracker.kinematic_state)
    npt.assert_allclose(round_tripped.covariance, tracker.covariance)
    npt.assert_allclose(round_tripped.shape_state, tracker.shape_state)
    npt.assert_allclose(round_tripped.shape_covariance, tracker.shape_covariance)


def test_append_and_flush_emit_fixed_lag_sequence():
    smoother = FixedLagMEMQKFSmoother(lag=1)
    first = _state(
        [0.0, 0.0, 1.0, 0.0],
        diag(array([0.5, 0.5, 0.5, 0.5])),
        [0.1, 2.0, 1.0],
        diag(array([0.2, 0.2, 0.2])),
    )
    second = _state(
        [1.0, -1.0, 3.0, 0.0],
        diag(array([0.4, 0.4, 0.4, 0.4])),
        [0.5, 3.0, 2.0],
        diag(array([0.1, 0.1, 0.1])),
    )
    predicted_second = _state(
        [0.0, 0.0, 1.0, 0.0],
        diag(array([1.0, 1.0, 1.0, 1.0])),
        [0.1, 2.0, 1.0],
        diag(array([0.4, 0.4, 0.4])),
    )

    assert smoother.append(first) is None
    emitted = smoother.append(predicted_state=predicted_second, filtered_state=second)
    assert emitted is not None
    npt.assert_allclose(emitted.kinematic_state, array([0.5, -0.5, 2.0, 0.0]))

    remaining = smoother.flush()
    assert len(remaining) == 1
    npt.assert_allclose(remaining[0].kinematic_state, second.kinematic_state)


def test_fixed_interval_smoother_runs_full_suffix_windows():
    smoother = FixedIntervalMEMQKFSmoother()
    filtered_states = [
        _state([0.0], array([[0.5]]), [0.1, 2.0, 1.0], diag(array([0.2, 0.2, 0.2]))),
        _state([1.0], array([[0.4]]), [0.3, 3.0, 2.0], diag(array([0.1, 0.1, 0.1]))),
        _state([2.0], array([[0.3]]), [0.4, 4.0, 3.0], diag(array([0.1, 0.1, 0.1]))),
    ]
    predicted_states = [
        _state([0.0], array([[1.0]]), [0.1, 2.0, 1.0], diag(array([0.4, 0.4, 0.4]))),
        _state([1.0], array([[1.0]]), [0.3, 3.0, 2.0], diag(array([0.4, 0.4, 0.4]))),
    ]

    smoothed_states, smoother_gains = smoother.smooth(
        filtered_states,
        predicted_states,
        system_matrices=array([[[1.0]], [[1.0]]]),
        shape_system_matrices=array([eye(3), eye(3)]),
    )

    assert len(smoothed_states) == 3
    assert len(smoother_gains) == 3
    assert len(smoother_gains[0]) == 2
    assert len(smoother_gains[1]) == 1
    assert smoother_gains[2] == []
    npt.assert_allclose(
        smoothed_states[-1].kinematic_state, filtered_states[-1].kinematic_state
    )


def test_fbfb_mem_qkf_conditions_shape_update_on_smoothed_kinematics():
    smoother = ForwardBackwardForwardBackwardMEMQKFSmoother(shape_smoothing="none")
    filtered_states = [
        _state(
            [0.0, 0.0],
            0.5 * eye(2),
            [0.0, 1.0, 1.0],
            diag(array([0.2, 0.2, 0.2])),
        ),
        _state(
            [2.0, 0.0],
            0.5 * eye(2),
            [0.0, 1.0, 1.0],
            diag(array([0.2, 0.2, 0.2])),
        ),
    ]
    predicted_states = [
        _state(
            [0.0, 0.0],
            eye(2),
            [0.0, 1.0, 1.0],
            diag(array([0.4, 0.4, 0.4])),
        )
    ]
    measurements = [
        array([[2.0], [0.0]]),
        array([[3.0], [0.0]]),
    ]
    meas_noise_cov = 0.05 * eye(2)

    smoothed_states, smoother_gains = smoother.smooth(
        filtered_states,
        predicted_states,
        measurements=measurements,
        system_matrices=eye(2),
        shape_system_matrices=eye(3),
        meas_noise_covs=meas_noise_cov,
        initial_shape_state=array([0.0, 1.0, 1.0]),
        initial_shape_covariance=diag(array([0.2, 0.2, 0.2])),
    )

    manual_prior = MEMQKFTrackerState(
        array([1.0, 0.0]),
        0.375 * eye(2),
        array([0.0, 1.0, 1.0]),
        diag(array([0.2, 0.2, 0.2])),
    )
    manual_tracker = manual_prior.to_tracker()
    measurement_matrix = eye(2)
    center_estimate = measurement_matrix @ manual_tracker.kinematic_state
    shape_measurement_covariance = (
        meas_noise_cov
        + measurement_matrix @ manual_tracker.covariance @ measurement_matrix.T
        + manual_tracker.axis_covariance
    )
    # pylint: disable-next=protected-access
    manual_tracker._update_single_measurement_qkf(
        measurements[0][:, 0],
        center_estimate,
        measurement_matrix,
        meas_noise_cov,
        manual_tracker.multiplicative_noise_cov,
        shape_measurement_covariance,
        update_kinematics=False,
    )

    npt.assert_allclose(smoothed_states[0].kinematic_state, array([1.0, 0.0]))
    npt.assert_allclose(smoothed_states[0].covariance, 0.375 * eye(2))
    npt.assert_allclose(smoothed_states[0].shape_state, manual_tracker.shape_state)
    npt.assert_allclose(
        smoothed_states[0].shape_covariance, manual_tracker.shape_covariance
    )
    assert smoother_gains[0][0].kinematic is not None
    assert smoother_gains[0][0].shape is None


def test_fbfb_mem_qkf_final_backward_pass_smooths_reconditioned_shape():
    smoother = ForwardBackwardForwardBackwardMEMQKFSmoother()
    filtered_states = [
        _state(
            [0.0, 0.0],
            0.5 * eye(2),
            [0.0, 1.0, 1.0],
            diag(array([0.2, 0.2, 0.2])),
        ),
        _state(
            [2.0, 0.0],
            0.5 * eye(2),
            [0.4, 1.5, 1.2],
            diag(array([0.1, 0.1, 0.1])),
        ),
    ]
    predicted_states = [
        _state(
            [0.0, 0.0],
            eye(2),
            [0.0, 1.0, 1.0],
            diag(array([0.5, 0.5, 0.5])),
        )
    ]

    smoothed_states, smoother_gains = smoother.smooth(
        filtered_states,
        predicted_states,
        measurements=[array([[2.0], [0.0]]), array([[3.0], [0.0]])],
        system_matrices=eye(2),
        shape_system_matrices=eye(3),
        meas_noise_covs=0.05 * eye(2),
        initial_shape_state=array([0.0, 1.0, 1.0]),
        initial_shape_covariance=diag(array([0.2, 0.2, 0.2])),
    )

    assert len(smoothed_states) == 2
    assert len(smoother_gains[0]) == 1
    assert smoother_gains[0][0].kinematic is not None
    assert smoother_gains[0][0].shape is not None


def test_fbfb_mem_qkf_extra_iterations_refilter_kinematics_with_smoothed_shape():
    filtered_states = [
        _state(
            [0.0, 0.0],
            0.5 * eye(2),
            [0.0, 1.0, 1.0],
            diag(array([0.2, 0.2, 0.2])),
        ),
        _state(
            [2.0, 0.0],
            0.5 * eye(2),
            [0.3, 1.4, 1.0],
            diag(array([0.15, 0.15, 0.15])),
        ),
    ]
    predicted_states = [
        _state(
            [0.0, 0.0],
            eye(2),
            [0.0, 1.0, 1.0],
            diag(array([0.4, 0.4, 0.4])),
        )
    ]
    measurements = [array([[2.0], [0.0]]), array([[4.0], [0.0]])]
    common_kwargs = {
        "filtered_states": filtered_states,
        "predicted_states": predicted_states,
        "measurements": measurements,
        "system_matrices": eye(2),
        "shape_system_matrices": eye(3),
        "meas_noise_covs": 0.05 * eye(2),
        "initial_shape_state": array([0.0, 1.0, 1.0]),
        "initial_shape_covariance": diag(array([0.2, 0.2, 0.2])),
    }

    one_pass_states, _ = ForwardBackwardForwardBackwardMEMQKFSmoother(
        n_iterations=1
    ).smooth(**common_kwargs)
    three_pass_states, three_pass_gains = ForwardBackwardForwardBackwardMEMQKFSmoother(
        n_iterations=3
    ).smooth(**common_kwargs)

    assert len(three_pass_states) == len(filtered_states)
    assert len(three_pass_gains[0]) == 1
    assert not np.allclose(
        np.asarray(three_pass_states[1].kinematic_state),
        np.asarray(one_pass_states[1].kinematic_state),
    )
    assert abs(float(three_pass_states[1].kinematic_state[0]) - 4.0) < abs(
        float(one_pass_states[1].kinematic_state[0]) - 4.0
    )
