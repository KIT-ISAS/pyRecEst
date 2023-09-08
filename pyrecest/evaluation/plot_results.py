import numpy as np
import matplotlib.pyplot as plt
import warnings
from .determine_all_deviations import determine_all_deviations
from .get_axis_label import get_axis_label
from .get_extract_mean import get_extract_mean
from .get_distance_function import get_distance_function

def plot_results(filename=None, plot_log=np.array([[True, True]]), plot_stds=False,
                omit_slow = True):
    """
    Plot results of filter evaluations.

    Args:
        filenames (str or list): File names to load results from.
        plot_log (np.array of bool): Whether to plot in log scale for each axis.
        plot_stds (bool): Whether to plot standard deviations.
        omit_slow (bool): Whether to omit slow filters from plotting.

    Returns:
        param_time_and_error_per_filter (dict): Dictionary containing performance statistics.
    """

    param_time_and_error_per_filter = {}
    not_all_warnings_shown = True if warnings.filters != 'always' else False

    if not_all_warnings_shown:
        print("Not all warnings are enabled.")

    # Expand plot_log to handle all plots (pad it)
    plot_log = np.pad(plot_log, ((False, False), (0, 3 - plot_log.shape[1])), 'constant')
    plot_random_filter = True
    
    if filename is None:
        # Prompt for filename
        filename = input("Please enter the filename to load: ")
        
    data = np.load(filename, allow_pickle=True).item()
    filter_results = data['last_filter_states']
    groundtruths = data['groundtruths'][0]
    scenario_param = data['scenario_param']


    if not plot_random_filter:
        filter_results = [res for res in filter_results if res['filter_name'] != 'random']

    if groundtruths.shape[0] < 1000:
        warnings.warn('Using less than 1000 runs. This may lead to unreliable results.')

    used_filter_names_short = [config["name"] for config in data['filter_configs']]    
    used_filter_names_long = get_used_filter_names_long(used_filter_names_short)
    
    
    axis_label = get_axis_label(scenario_param['manifold_type'])
    extract_mean = get_extract_mean(scenario_param['manifold_type'])
    distance_function = get_distance_function(scenario_param['manifold_type'])
    
    all_errors = determine_all_deviations(filter_results, extract_mean, distance_function, groundtruths)
    
    for i, curr_filter_name in enumerate(used_filter_names_short):
        # Iterate over all possible names and plot the lines for those that were evaluated
        color, style_marker, style_line = get_plot_style_for_filter(curr_filter_name)

        is_correct_filter = [curr_filter_name == filter_name for filter_name in used_filter_names_short]
        params_sorted = np.array([r.filterParams for r in filter_results if r.filterName == data['filter_configs']])
        order = np.argsort(params_sorted)
        if curr_filter_name.startswith('ff') or curr_filter_name == 'htgf':
            params_sorted = params_sorted ** groundtruths[0].shape[0]
        elif 'shf' in curr_filter_name:
            params_sorted = (params_sorted + 1) ** 2

        rmses = np.array([e for i, e in enumerate(all_errors) if is_correct_filter[i]])
        rmsesSorted = rmses[order]
        stds = np.array([s for i, s in enumerate(all_stds) if is_correct_filter[i]])
        stdsSorted = stds[order]
        times = np.array([t for i, t in enumerate(all_mean_times) if is_correct_filter[i]])
        timesSortedAndScaled = times[order] * times_factor

        if not np.any(np.isnan(params_sorted)) and len(params_sorted) > 1:
            if plot_stds:
                plt.plot(params_sorted, rmses_sorted - stds_sorted * 0.05, color=color)
                plt.plot(params_sorted, rmses_sorted + stds_sorted * 0.05, color=color)
            handles_error_over_param.append(plt.plot(params_sorted, rmses_sorted, style_marker + style_line, color=color)[0])
        else:  # If not parametric like wn or twn filter
            handles_error_over_param.append(plt.plot([min_param, max_param], [rmses_sorted, rmses_sorted], style_line, color=color)[0])

        if not np.any(np.isnan(params_sorted)) and len(params_sorted) > 1:
            handles_time_over_param.append(plt.plot(params_sorted, times_sorted_and_scaled, style_marker + style_line, color=color)[0])
        else:
            if omit_slow and np.max(all_mean_times[~is_correct_filter]) < times_sorted_and_scaled / 2:
                handles_time_over_param.append(None)
                print(f"{curr_filter_name} took very long ({times_sorted_and_scaled} ms), not plotting it for additional clarity")
            else:
                handles_time_over_param.append(plt.plot([min_param, max_param], [times_sorted_and_scaled, times_sorted_and_scaled], style_line, color=color)[0])

        if not np.any(np.isnan(params_sorted)) and len(params_sorted) > 1:
            handles_error_over_time.append(plt.plot(times_sorted_and_scaled, rmses_sorted, style_marker + style_line, color=color)[0])
        else:
            if omit_slow and np.max(all_mean_times[~is_correct_filter]) < times_sorted_and_scaled / 1.5:
                handles_error_over_time.append(None)
                print(f"{curr_filter_name} took very long ({times_sorted_and_scaled} ms), not plotting it for additional clarity")
            else:
                handles_error_over_time.append(plt.plot(times_sorted_and_scaled, rmses_sorted, style_marker, color=color)[0])
        # Results for MTT
        # Store results for filter
        index_for_struct = [i for i, f in enumerate(param_time_and_error_per_filter) if f.filterName == curr_filter_name][0]
        param_time_and_error_per_filter[index_for_struct].all_params = params_sorted
        param_time_and_error_per_filter[index_for_struct].mean_times_all_configs = times_sorted_and_scaled
        param_time_and_error_per_filter[index_for_struct].mean_error_all_configs = rmses_sorted
    
    if not_all_warnings_shown:
        print("-----------Reminder: Not all warnings were enabled.-----------")

    return param_time_and_error_per_filter

def get_plot_style_for_filter(filter_name):
    """
    Get plot style for a given filter name.

    Args:
        filter_name (str): Name of the filter.

    Returns:
        color (str or list): Color for the plot.
        style_marker (str): Marker style for the plot.
        style_line (str): Line style for the plot.
    """

    switcher = {
        'iff': ([0.4660, 0.6740, 0.1880], 'o', '--'),
        'ishf': ([0.4660, 0.6740, 0.1880], 'o', '--'),
        'discrete': ([0.8500, 0.3250, 0.0980], '', '-'),
        'bingham': ([0.8500, 0.3250, 0.0980], '', '-'),
        'sqff': ([0, 0.4470, 0.7410], 'd', '-'),
        'sqshf': ([0, 0.4470, 0.7410], 'd', '-'),
        'pf': ([0.9290, 0.6940, 0.1250], 'p', '-.'),
        'htpf': ([0.9290, 0.6940, 0.1250], 'p', '-.'),
        'wn': ([0.4940, 0.1840, 0.5560], 'd', '-'),
        'vm': ([0.4940, 0.1840, 0.5560], 'd', '-'),
        'vmf': ([0.4940, 0.1840, 0.5560], 'd', '-'),
        'hhgf': ([0.4940, 0.1840, 0.5560], 'd', '-'),
        'se2bf': ([0.4940, 0.1840, 0.5560], 'd', '-'),
        'random': ([0, 0, 0], 'x', '-'),
        'dummy': ([0, 0, 0], 'x', '-'),
        'kf': ([0.3010, 0.7450, 0.9330], '*', '-'),
        'se2iukf': ([0.3010, 0.7450, 0.9330], '*', '-'),
        'hgfSymm': ([0.3010, 0.7450, 0.9330], '', '-'),
        'figResetOnPred': ([0.3010, 0.7450, 0.9330], '', '-'),
        'sgf': ([0.6350, 0.0780, 0.1840], '*', '-'),
        'hgf': ([0.6350, 0.0780, 0.1840], '*', '-'),
        's3f': ([0.6350, 0.0780, 0.1840], '*', '-'),
        'fig': ([0.6350, 0.0780, 0.1840], '*', '-'),
        'htgf': ([0.6350, 0.0780, 0.1840], '*', '-.'),
    }

    color, style_marker, style_line = switcher.get(filter_name, ('k', '', '-'))

    return color, style_marker, style_line

def get_used_filter_names_long(used_filter_names):
    supported_filters_short_names = ['se2ukfm', 'se2bf', 's3f', 'grid', 'iff', 'sqff', 'pf', 'htpf', 'vmf', 'bingham', 'wn', 'vm', 'twn', 'kf', 'ishf', 'sqshf', 'htgf', 'sgf', 'hgf', 'hgfSymm', 'hhgf', 'randomTorus', 'randomSphere', 'fig', 'figResetOnPred', 'gnn', 'GNN']
    supported_filters_long_names = ['Unscented KF for Manifolds', '(Progressive) SE(2) Bingham filter', 'State space subdivision filter', 'Grid filter', 'Fourier identity filter', 'Fourier square root filter', 'Particle filter', 'Particle filter', 'Von Mises--Fisher filter', 'Bingham filter', 'Wrapped normal filter', 'Von Mises filter', 'Bivariate WN filter', 'Kalman filter', 'Spherical hamonics identity filter', 'Spherical hamonics square root filter', 'Hypertoroidal grid filter', 'Spherical grid filter', 'Hyperspherical grid filter', 'Symmetric hyperspherical grid filter', 'Hyperhemispherical grid filter', 'Random filter', 'Random filter', 'Fourier-interpreted grid filter', 'FIG-Filter with resetting on prediction', 'Global Nearest Neighbor', 'Global Nearest Neighbor']

    used_filter_names_long = [supported_filters_long_names[supported_filters_short_names.index(name)] if name in supported_filters_short_names else name for name in used_filter_names]

    return used_filter_names_long