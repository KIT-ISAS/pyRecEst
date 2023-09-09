import datetime
import os
from typing import Any

import numpy as np

from .check_and_fix_params import check_and_fix_params
from .iterate_configs_and_runs import iterate_configs_and_runs

# pylint: disable=R0913,R0914
def start_evaluation(
    scenario_param: str | dict[str, Any],
    filter_configs: list[dict[str, Any]],
    n_runs: int,
    save_folder: str = ".",
    plot_each_step: bool = False,
    convert_to_point_estimate_during_runtime: bool = False,
    extract_all_point_estimates: bool = False,
    tolerate_failure: bool = False,
    auto_warning_on_off: bool = False,
    ):
    scenario_param["plot"] = plot_each_step
    scenario_param = check_and_fix_params(scenario_param)

    filter_configs = [
        {"name": f["name"], "parameter": p}
        for f in filter_configs
        for p in (f["parameter"] or [None])
    ]
    
    (
        last_filter_states,  # pylint: disable=R0801
        runtimes,  # pylint: disable=R0801
        run_failed,  # pylint: disable=R0801
        groundtruths,  # pylint: disable=R0801
        measurements,  # pylint: disable=R0801
    ) = iterate_configs_and_runs(
        scenario_param,
        filter_configs,
        n_runs,
        convert_to_point_estimate_during_runtime,
        extract_all_point_estimates,
        tolerate_failure,
        auto_warning_on_off,
    )

    date_and_time = datetime.datetime.now()
    filename = os.path.join(
        save_folder,
        f"{scenario_param['name']}_{date_and_time.strftime('%Y-%m-%d--%H-%M-%S')}.npy",
    )
    np.save(
        filename,
        {
            "groundtruths": groundtruths,
            "measurements": measurements,
            "run_failed": run_failed,
            "last_filter_states": last_filter_states,
            "runtimes": runtimes,
            "scenario_param": scenario_param,
            "filter_configs": filter_configs,
        },
        allow_pickle=True,
    )

    return (
        scenario_param,  # pylint: disable=R0801
        filter_configs,  # pylint: disable=R0801
        last_filter_states,  # pylint: disable=R0801
        runtimes,  # pylint: disable=R0801
        run_failed,  # pylint: disable=R0801
        groundtruths,  # pylint: disable=R0801
        measurements,  # pylint: disable=R0801
    )