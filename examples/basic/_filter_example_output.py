"""Shared output helpers for basic filter examples."""


def _format_position_velocity_row(step, measurement, estimate):
    """Return one formatted position/velocity estimate row."""
    position, velocity = estimate
    return (
        f"{step:>4}  {measurement:>11.2f}  "
        f"{float(position):>8.3f}  {float(velocity):>8.3f}"
    )


def print_position_velocity_estimates(
    measurements,
    estimates,
    *,
    final_covariance=None,
):
    """Print a compact position/velocity estimate table."""
    print("step  measurement  position  velocity")
    rows = (
        _format_position_velocity_row(index + 1, measurement, estimates[index])
        for index, measurement in enumerate(measurements)
    )
    print("\n".join(rows))

    if final_covariance is not None:
        print("\nFinal covariance:")
        print(final_covariance)
