from typing import Any

import numpy as np
import os

from .evaluate_for_variables import evaluate_for_variables


# pylint: disable=R0913,R0914
def evaluate_for_file(
    input_file_name: str,
    filter_configs: list[dict[str, Any]],
    manifold,
    save_folder: str = ".",
    plot_each_step: bool = False,
    convert_to_point_estimate_during_runtime: bool = False,
    extract_all_point_estimates: bool = False,
    tolerate_failure: bool = False,
    auto_warning_on_off: bool = False,
) -> tuple[
    dict,
    list[dict],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray[np.ndarray],
]:
    data = np.load(input_file_name, allow_pickle=True).item()
    
    scenario_name = os.path.splitext(os.path.basename(input_file_name))[0]

    scenario_config = {"name": scenario_name, "manifold": manifold}

    return evaluate_for_variables(data["groundtruths"], data["measurements"], filter_configs,
                           scenario_config,
                           save_folder,
                           plot_each_step, convert_to_point_estimate_during_runtime,
                           extract_all_point_estimates,
                           tolerate_failure, auto_warning_on_off)
