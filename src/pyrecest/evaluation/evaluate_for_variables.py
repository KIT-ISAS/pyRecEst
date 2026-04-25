import datetime
import os
from typing import Any

import numpy as np

from .iterate_configs_and_runs import iterate_configs_and_runs


def _filter_parameter_values(parameter):
    if parameter is None:
        return [None]
    if isinstance(parameter, (list, tuple)):
        return list(parameter) or [None]
    return [parameter]


def _expand_filter_configs(filter_configs):
    return [
        {"name": filter_config["name"], "parameter": parameter}
        for filter_config in filter_configs
        for parameter in _filter_parameter_values(filter_config["parameter"])
    ]


# pylint: disable=R0913,R0914, too-many-positional-arguments
def evaluate_for_variables(
    groundtruths,
    measurements,
    filter_configs: list[dict[str, Any]],
    # Dict with simulation config (for simulated scenarios) or other information (for scenarios read from file)
    scenario_config=None,
    save_folder: str = ".",
    plot_each_step: bool = False,
    convert_to_point_estimate_during_runtime: bool = False,
    extract_all_point_estimates: bool = False,
    tolerate_failure: bool = False,
    auto_warning_on_off: bool = False,
):
    if scenario_config is None:
        scenario_config = {"name": "custom"}

    evaluation_config = {
        "plot_each_step": plot_each_step,
        "convert_to_point_estimate_during_runtime": convert_to_point_estimate_during_runtime,
        "extract_all_point_estimates": extract_all_point_estimates,
        "tolerate_failure": tolerate_failure,
        "auto_warning_on_off": auto_warning_on_off,
    }

    filter_configs = _expand_filter_configs(filter_configs)

    (
        last_filter_states,  # pylint: disable=R0801
        runtimes,  # pylint: disable=R0801
        run_failed,  # pylint: disable=R0801
        groundtruths,  # pylint: disable=R0801
        measurements,  # pylint: disable=R0801
    ) = iterate_configs_and_runs(
        groundtruths, measurements, scenario_config, filter_configs, evaluation_config
    )

    date_and_time = datetime.datetime.now()
    filename = os.path.join(
        save_folder,
        f"{scenario_config['name']}_{date_and_time.strftime('%Y-%m-%d--%H-%M-%S')}.npy",
    )

    np.save(
        filename,
        {
            "groundtruths": groundtruths,
            "measurements": measurements,
            "run_failed": run_failed,
            "last_filter_states": last_filter_states,
            "runtimes": runtimes,
            "scenario_config": scenario_config,
            "filter_configs": filter_configs,
            "evaluation_config": evaluation_config,
        },
        allow_pickle=True,
    )

    return (
        last_filter_states,  # pylint: disable=R0801
        runtimes,  # pylint: disable=R0801
        run_failed,  # pylint: disable=R0801
        groundtruths,  # pylint: disable=R0801
        measurements,  # pylint: disable=R0801
        scenario_config,  # pylint: disable=R0801
        filter_configs,  # pylint: disable=R0801
        evaluation_config,  # pylint: disable=R0801
    )
