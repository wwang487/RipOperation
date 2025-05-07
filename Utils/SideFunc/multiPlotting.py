import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns
import copy
from scipy.stats import kde, gaussian_kde
from scipy.optimize import curve_fit
from scipy.interpolate import griddata
from matplotlib.colors import to_rgba
from matplotlib.colors import Normalize, ListedColormap, LinearSegmentedColormap
from matplotlib.colorbar import ColorbarBase
from matplotlib.cm import get_cmap
import matplotlib.patches as patches
from matplotlib.legend_handler import HandlerPatch
from datetime import datetime, timedelta
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from scipy.spatial import ConvexHull
from scipy.interpolate import interp1d
from matplotlib.patches import Polygon
from mpl_toolkits.basemap import Basemap
import contextily as ctx
from matplotlib.lines import Line2D
import ternary
import matplotlib.path as mpath
from matplotlib.dates import DateFormatter, DayLocator
import pandas as pd
from dateutil.parser import parse
from dateutil.relativedelta import relativedelta
from matplotlib import colormaps
from matplotlib.ticker import MultipleLocator
from matplotlib.font_manager import FontProperties
from matplotlib.cm import ScalarMappable
import statsmodels.api as sm
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker

class __HandlerRect(HandlerPatch):
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        # Customize the size and location of the rectangle for the legend symbol
        center = 0.5 * width - 0.5 * xdescent, 0.5 * ydescent
        p = patches.Rectangle(xy=center, width=width * 0.6, height=height * 1.5)  # Custom size
        self.update_prop(p, orig_handle, legend)
        p.set_transform(trans)
        return [p]

def __get_time_bounds(data):
    """
    Returns the earliest start time and the latest end time from a list of time intervals.
    """
    start_times = [parse(interval[0]) if isinstance(interval[0], str) else interval[0] for interval in data]
    end_times = [parse(interval[1]) if isinstance(interval[1], str) else interval[1] for interval in data]
    overall_start = min(start_times)
    overall_end = max(end_times)
    return overall_start, overall_end

def __generate_tick_times(data, time_resolution='y', time_interval=1):
    """
    Generates a list of datetime objects for ticks based on the overall time range, interval, and resolution.
    """
    overall_start, overall_end = __get_time_bounds(data)
    
    # Define the correct frequency string and adjustment parameters for relativedelta
    freq_map = {
        'y': 'AS',  # Start of each year
        'm': 'MS',  # Start of each month
        'd': 'D',   # Each day
        'H': 'H',   # Each hour
        'M': 'T',   # Each minute
        'S': 'S'    # Each second
    }
    # Translate time resolution to relativedelta compatible terms
    delta_map = {
        'y': 'years',
        'm': 'months',
        'd': 'days',
        'H': 'hours',
        'M': 'minutes',
        'S': 'seconds'
    }
    freq_str = str(time_interval) + freq_map[time_resolution]
    
    # Extend the date range to improve plot appearance
    adjustment_kwargs = {delta_map[time_resolution]: time_interval}
    overall_start -= relativedelta(**adjustment_kwargs)
    overall_end += relativedelta(**adjustment_kwargs)
    
    ticks = pd.date_range(start=overall_start, end=overall_end, freq=freq_str)
    return ticks

def __create_transparent_cmap(cmap, alpha=0.5):
    """Create a transparent version of the given colormap."""
    colors = cmap(np.arange(cmap.N))
    colors[:, -1] = alpha  # Set the alpha channel
    return LinearSegmentedColormap.from_list(cmap.name + "_alpha", colors)
def __get_bar_data(input_list, bin_tuple_list):
    # bin_tuple_list is a list of tuples, each tuple contains the start and end of a bin
    res = []
    for i in range(len(bin_tuple_list)):
        if i == 0:
            end = bin_tuple_list[i]
            start = -999
        elif i == len(bin_tuple_list) - 1:
            start = bin_tuple_list[i]
            end = 999999
        else:
            start, end = bin_tuple_list[i][0], bin_tuple_list[i][1]
        count = 0
        for j in range(len(input_list)):
            if start <= input_list[j] < end:
                count += 1
        res.append(count)
    return res

def __find_bin_ind(val, bin_ticks, min_val = 0, max_val = 999999):
    if min_val <= val < bin_ticks[0]:
        return 0
    elif val >= bin_ticks[-1]:
        return len(bin_ticks)
    else:
        for i in range(len(bin_ticks) - 1):
            temp_tick = bin_ticks[i]
            if temp_tick <= val < bin_ticks[i + 1]:
                return i + 1
    

def __analyze_cross_relationship(data, key1, key2, key1_ticks = None, key2_ticks = None, x_scale = 1, y_scale = 1):
    """
    Analyzes the cross-relationship between two variables in a list of dictionaries.

    :param data: List of dictionaries containing the data.
    :param key1: The first key in the dictionaries to analyze.
    :param key2: The second key in the dictionaries to analyze.
    :return: A dictionary with keys as tuple pairs of values from key1 and key2, and values as their frequency.
    """
    pair_frequency = {}
    if isinstance(data, pd.DataFrame):
        for index, row in data.iterrows():
            # Extract the values associated with key1 and key2
            value1 = row.get(key1) / x_scale
            value2 = row.get(key2) / y_scale
            
            # Skip entries where either key is missing
            if value1 is None or value2 is None:
                continue
            
            # Create a tuple from the two values
            if key1_ticks is None or key2_ticks is None:
                key_pair = (value1, value2)
            else:
                key_pair = (__find_bin_ind(value1, key1_ticks), __find_bin_ind(value2, key2_ticks))
            if key_pair[0] is None:
                print(value1)
            # Increment the count for this key pair in the dictionary
            if key_pair in pair_frequency:
                pair_frequency[key_pair] += 1
            else:
                pair_frequency[key_pair] = 1
    else:
        for entry in data:
            #print(entry)
            # Extract the values associated with key1 and key2
            value1 = entry.get(key1) / x_scale
            value2 = entry.get(key2) / y_scale
            
            # Skip entries where either key is missing
            if value1 is None or value2 is None:
                continue
            
            # Create a tuple from the two values
            if key1_ticks is None or key2_ticks is None:
                key_pair = (value1, value2)
            else:
                key_pair = (__find_bin_ind(value1, key1_ticks), __find_bin_ind(value2, key2_ticks))
            if key_pair[0] is None:
                print(value1)
            # Increment the count for this key pair in the dictionary
            if key_pair in pair_frequency:
                pair_frequency[key_pair] += 1
            else:
                pair_frequency[key_pair] = 1

    return pair_frequency

def __generate_ticklabels(tick_bins, is_close_1, is_close_2):
        res = []
        if not is_close_1:
            res.append(f'<{tick_bins[0]}')
        for i in range(len(tick_bins) - 1):
            res.append(f'{tick_bins[i]}-{tick_bins[i + 1]}')
        if not is_close_2:
            res.append(f'>{tick_bins[-1]}')
        return res

def __process_text_labels(orig_label, sep='_'):
    lowercase_words = {'of', 'at', 'on', 'in', 'to', 'for', 'with', 'a', 'an', 'the', 'and', 'but', 'or', 'nor', 'so', 'yet', 'against'}
    
    # Split the sentence into words
    words = orig_label.split(sep)
    
    # Capitalize the first word and others based on their position and whether they are in the lowercase_words set
    title_cased_words = [words[0].capitalize()] + [word.capitalize() if word.lower() not in lowercase_words else word for word in words[1:]]
    
    # Join the words back into a sentence
    return ' '.join(title_cased_words)

def __generate_bin_ticks(data, num_bins, mode='data', smart_round=False):
    """
    Generate bin ticks based on percentiles or range for given data, with optional generalized smart rounding.
    
    Args:
    data (sequence): A sequence of numeric data (list, tuple, numpy array, etc.).
    num_bins (int): The number of bins to divide the data into.
    mode (str): 'data' for percentile-based bins, 'range' for evenly spaced bins.
    smart_round (bool): Apply generalized smart rounding to the bin edges based on their magnitude.
    
    Returns:
    np.array: An array containing the bin edges.
    """
    if not isinstance(data, np.ndarray):
        data = np.array(data)  # Convert data to numpy array if not already
    
    if mode == 'data':
        percentiles = np.linspace(0, 100, num_bins + 1)
        bin_edges = np.percentile(data, percentiles)
    elif mode == 'range':
        min_val, max_val = np.min(data), np.max(data)
        bin_edges = np.linspace(min_val, max_val, num_bins + 1)
    else:
        raise ValueError("Mode must be 'data' or 'range'")

    if smart_round:
        bin_edges = np.vectorize(__smart_rounding)(bin_edges)
    
    return bin_edges   

def __smart_rounding(value):
    """
    Dynamically round the value based on its magnitude.
    
    Args:
    value (float): The value to be rounded.
    
    Returns:
    float: The rounded value.
    """
    if value == 0:
        return 0
    magnitude = int(np.floor(np.log10(abs(value))))
    if magnitude < -2:
        return round(value, -magnitude + 1)
    elif magnitude < 0:
        return round(value, -magnitude)
    else:
        power = 10 ** magnitude
        return round(value / power) * power
    
def __generate_levels(data, num_levels, density_threshold = None):
    """
    Generate contour levels based on the data and number of levels.
    """
    if density_threshold == None:
        levels = np.linspace(data.min(), data.max(), num_levels)
    else:
        bottom_thresh = max(data.min(), density_threshold)
        levels = np.linspace(bottom_thresh, data.max(), num_levels - 1)
    return levels

def __find_tick_digits(input_val):
    if input_val == 0:
        return 0
    magnitude = int(np.floor(np.log10(abs(input_val))))
    return magnitude
    
def __round_to_nearest(number, power):
    factor = 10 ** power
    return round(number / factor) * factor

def __scale(ax, bottom, scale_lim, theta, width, scale_major):
    t = np.linspace(theta-width/2, theta+width/2, 6)
    for i in range(int(bottom), int(bottom+scale_lim+scale_major), scale_major):
        ax.plot(t, [i]*6, linewidth=0.25, color='gray', alpha=0.8)

def __scale_value(ax, bottom, theta, scale_lim, scale_major, tick_font_size = 5):
    for i in range(int(bottom), int(bottom+scale_lim+scale_major), scale_major):
        ax.text(theta, i, f'{i-bottom}',
                fontsize=tick_font_size, alpha=0.8,
                va='center',  ha='center')

def plot_beeswarm(categories, values, optimizer_value=None, desired_value=None, fig_size=(12, 8), swarm_color='gray', swarm_size=2,
                  swarm_transparency=None, median_color='black', median_edge_color='none', median_size=30,
                  line_color='black', line_thickness=2, optimizer_color='black', optimizer_style='--', optimizer_thickness=1.5,
                  desired_color='blue', desired_style='-', desired_thickness=1.5, title='Subcatchment Area Distribution by Site',
                  xlabel_name='Category', ylabel_name='Subcatchment Area (km²)', xlabel_size=12, ylabel_size=12,
                  xlabel_font='Arial', ylabel_font='Arial', tick_font_name='Arial', tick_font_size=10, xtick_rotation=0,
                  y_range=None, is_show=True, save_path=None, optimizer_label='Optimizer Value',
                  desired_label='Desired Value'):
    """
    Plots a beeswarm plot with medians, interquartile ranges (IQR), and reference lines.

    :param categories: List of category labels (e.g., different sites or conditions).
    :param values: List of arrays, each containing values corresponding to a category.
    :param optimizer_bandwidth: The optimizer bandwidth to plot as a horizontal dashed line.
    :param desired_value: The desired value to plot as a horizontal solid line.
    """
    # Initialize the plot
    plt.figure(figsize=fig_size)
    
    # Create the beeswarm plot
    sns.swarmplot(data=values, size=swarm_size, color=swarm_color, edgecolor=median_edge_color, alpha=swarm_transparency)
    
    # Calculate and plot medians and IQRs
    for i, v in enumerate(values):
        median = np.median(v)
        quartile1 = np.percentile(v, 25)
        quartile3 = np.percentile(v, 75)
        
        # Plot median
        plt.scatter(i, median, color=median_color, s=median_size, zorder=3)
        
        # Plot vertical bar for the IQR
        plt.vlines(i, quartile1, quartile3, color=line_color, linestyle='-', lw=line_thickness)
    
    # Add optimizer bandwidth line if provided
    if optimizer_value is not None:
        plt.axhline(y=optimizer_value, color=optimizer_color, linestyle=optimizer_style, linewidth=optimizer_thickness, label=optimizer_label)
    
    # Add desired value line if provided
    if desired_value is not None:
        plt.axhline(y=desired_value, color=desired_color, linestyle=desired_style, linewidth=desired_thickness, label=desired_label)
    
    # Setting labels and titles
    plt.xticks(ticks=np.arange(len(categories)), labels=categories, rotation=xtick_rotation, fontsize=tick_font_size, fontname=tick_font_name)
    plt.xlabel(xlabel_name, color='black', fontsize=xlabel_size, fontname=xlabel_font)
    plt.ylabel(ylabel_name, color='black', fontsize=ylabel_size, fontname=ylabel_font)
    plt.title(title)
    
    if y_range:
        plt.ylim(y_range)
    else:
        min_val, max_val = min([min(v) for v in values]), max([max(v) for v in values])
        delta = max_val - min_val
        plt.ylim(min_val - 0.1 * delta, max_val + 0.15 * delta)
    
    # Display legend only if any line is plotted
    if optimizer_value is not None or desired_value is not None:
        plt.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='black', fontsize=10)
    
    if save_path:
        plt.savefig(save_path)
    
    if is_show:
        plt.tight_layout()
        plt.show()


def plot_bar_plots(list_of_lists, tuple_range_list, titles = '', ylabels='', bar_color='blue', bar_edgecolor='black', fig_size=(10, 6), tick_fontname='Arial',
                    tick_fontsize=12, title_fontsize=14, label_fontsize=14, line_color='red', show_all_xticklabels=True, bar_width = 1,
                    line_style='--', line_width=2, is_legend=False, unit='m', is_fixed_y_range=True, y_range=[0, 20], is_mean_value = False,
                    is_scaled=False, scale_factor=10, save_path='', is_show=False, is_save=True, transparent_bg=True, horizontal = True, 
                    convert_minute = True, hspace=0.05):
    '''
    This function is used to plot multiple bar plots in one figure. The input data should be a list of lists, where each list contains the data for one bar plot.
    
    Parameters:
    list_of_lists: list of lists, the input data for the bar plots
    tuple_range_list: list of tuples, the range of each bar
    titles: str or list of str, the title of each bar plot, if it is empty, no title will be shown
    ylabels: str or list of str, the y label of each bar plot, if it is empty, no y label will be shown
    bar_color: str, the color of the bars
    bar_edgecolor: str, the edge color of the bars
    fig_size: tuple, the size of the figure
    tick_fontname: str, the font name of the ticks
    tick_fontsize: int, the font size of the ticks
    title_fontsize: int, the font size of the titles
    label_fontsize: int, the font size of the labels
    line_color: str, the color of the mean line
    show_all_xticklabels: bool, whether to show all x tick labels
    bar_width: float, the width of the bars
    line_style: str, the style of the mean line
    line_width: float, the width of the mean line
    is_legend: bool, whether to show the legend
    unit: str, the unit of the data
    is_fixed_y_range: bool, whether to fix the y range
    y_range: list, the y range
    is_mean_value: bool, whether to show the mean value
    is_scaled: bool, whether to scale the data
    scale_factor: float, the scale factor
    save_path: str, the path to save the figure
    is_show: bool, whether to show the figure
    is_save: bool, whether to save the figure
    transparent_bg: bool, whether to set the background to be transparent
    horizontal: bool, whether to plot the bar horizontally
    convert_minute: bool, whether to convert the x tick labels to minutes
    hspace: float, the space between subplots
    
    Returns:
    None
    
    If you want to customize the plot, you can modify the code in this function.
    
    '''
    
    n = len(list_of_lists)
    w, h = fig_size
    
    if is_fixed_y_range and y_range is None:
        max_bar_value = 0
        for data in list_of_lists:
            if is_scaled:
                data = np.array(data) / scale_factor
            bars = __get_bar_data(data, tuple_range_list)
            max_bar_value = max(max_bar_value, bars.max())
        y_range = [0, max_bar_value * 1.05]
    
    fig, axs = plt.subplots(n, 1, figsize=(w, h * n))

    for i, data in enumerate(list_of_lists):
        if is_scaled:
            data = np.array(data) / scale_factor
        
        bar_positions = np.arange(len(tuple_range_list))
        bars = __get_bar_data(data, tuple_range_list)  # This function needs to be defined to get bar data
        bars = np.array(bars) / np.sum(bars) * 100
        if horizontal:
            axs[i].barh(bar_positions, bars, color=bar_color, edgecolor=bar_edgecolor)
        else:
            axs[i].bar(bar_positions, bars, color=bar_color, edgecolor=bar_edgecolor, width=bar_width)
        
        # Calculate and plot the mean line
        if is_mean_value:
            mean_value = np.mean(data)
            axs[i].axvline(mean_value, color=line_color, linestyle=line_style, linewidth=line_width, label=f'Mean: {mean_value:.2f} {unit}')
        
        temp_title = titles if titles == None or isinstance(titles, str) else titles[i]
        if temp_title:
            axs[i].set_title(temp_title, fontsize=title_fontsize, fontname=tick_fontname)
        
        x_tick_labels = []
        convert_factor = 1 if not convert_minute else 60
        for j in range(len(tuple_range_list)):
            if j == len(tuple_range_list) - 1:
                if tuple_range_list[j]/60 >= 1:
                    x_tick_labels.append(f'>{round(tuple_range_list[j]/convert_factor)}')
                else:
                    x_tick_labels.append(f'>{tuple_range_list[j]/convert_factor}')
            elif j == 0:
                if tuple_range_list[j]/60 >= 1:
                    x_tick_labels.append(f'<{round(tuple_range_list[j]/convert_factor)}')
                else:
                    x_tick_labels.append(f'<{tuple_range_list[j]/convert_factor}')
                
            else:
                if tuple_range_list[j][0]/60 >= 1:
                    x_tick_labels.append(f'{round(tuple_range_list[j][0]/convert_factor)}-{round(tuple_range_list[j][1]/convert_factor)}')
                elif tuple_range_list[j][1]/60 >= 1:
                    x_tick_labels.append(f'{tuple_range_list[j][0]/convert_factor}-{round(tuple_range_list[j][1]/convert_factor)}')
                else:
                    x_tick_labels.append(f'{tuple_range_list[j][0]/convert_factor}-{tuple_range_list[j][1]/convert_factor}')
        
        if horizontal:
            axs[i].set_yticks(bar_positions)
            axs[i].set_yticklabels(x_tick_labels,fontsize=tick_fontsize, fontname=tick_fontname)
            # Also needs to make the tick label orientation align with y
            axs[i].tick_params(axis='y', rotation=45)
        else:
            if i == len(list_of_lists) - 1:
                # last x label for each bar should be the range of tuple, also consider that the last tuple should be >, the first should be >
                axs[i].set_xticks(bar_positions)
                axs[i].set_xticklabels(x_tick_labels, fontsize=tick_fontsize, fontname=tick_fontname)
        if i < len(list_of_lists) - 1:
            axs[i].set_xticks([])
        
        if isinstance(ylabels, list) and ylabels[i]:
            axs[i].set_ylabel(ylabels[i], fontsize=label_fontsize, fontname=tick_fontname)
        
        if is_legend:
            axs[i].legend(loc="upper left")
        
        axs[i].grid(False)
        axs[i].tick_params(axis='both', which='major', labelsize=tick_fontsize)
        
        if not show_all_xticklabels and i != n - 1:
            axs[i].set_xticklabels([])
        if is_fixed_y_range:
            axs[i].set_ylim(y_range) if not horizontal else axs[i].set_xlim(y_range)
        if transparent_bg:
            axs[i].patch.set_alpha(0)

    plt.tight_layout()
    plt.subplots_adjust(hspace=hspace)
    if is_show:
        plt.show()
    if is_save:
        if save_path:
            fig.savefig(save_path, dpi=600, transparent=transparent_bg)
        else:
            print("Please provide a valid path to save the figure.")

def draw_cat_bar_curveplots(main_result, other_data_list, bar_colors=None, bar_thickness=0.8, bar_edge_color='black', line_color='black', cat_labels=None,
                       y_range=None, fig_size=(10, 6), xlabels=None, ylabels=None, line_thickness=1, tick_fontsize=10, tick_fontname='sans-serif', 
                       x_tick_interval=1, rotation=45, is_show=False, is_save=True, save_path='', is_legend=True, legend_size=12):
    '''
    This function is used to draw bar plots with multiple curve plots with a line plot for each dataset.
    
    Parameters:
    main_result: dict, the main result for the stacked bar plot
    other_data_list: list of dict, the other datasets for the line plots
    bar_colors: list, the colors for the bars
    bar_thickness: float, the thickness of the bars
    bar_edge_color: str, the edge color of the bars
    line_color: str, the color of the line plots
    cat_labels: list, the labels for each category
    xlabels: list, the labels for the x-axis
    ylabels: list, the labels for the y-axis
    y_range: list, the y range for each subplot
    fig_size: tuple, the size of the figure
    line_thickness: float or list, the thickness of the line plots
    tick_fontsize: int, the font size of the ticks
    tick_fontname: str, the font name of the ticks
    x_tick_interval: int, the interval of the x ticks
    rotation: int, rotation angle for the x-tick labels
    is_show: bool, whether to show the figure
    is_save: bool, whether to save the figure
    save_path: str, the path to save the figure
    
    Returns:
    None
    
    If you want to customize the plot, you can modify the code in this function.
    '''
    
    def prepare_data(result):
        dates = list(result.keys())
        values = list(result.values())
        return pd.DataFrame(values, index=pd.to_datetime(dates))
    
    def is_number(variable):
        return isinstance(variable, (int, float))

    main_df = prepare_data(main_result)
    all_series = [prepare_data(data) for data in other_data_list]

    fig, axes = plt.subplots(len(all_series) + 1, 1, figsize=fig_size, sharex=True)

    # If bar_colors are not provided, use a default color list
    if bar_colors is None:
        bar_colors = ['#377eb8', '#ff7f00', '#4daf4a', '#e41a1c', '#984ea3']

    # Plot the main result as a stacked bar plot
    bottom_series = None
    for i, col in enumerate(main_df.columns):
        color = bar_colors[i % len(bar_colors)]
        axes[0].bar(main_df.index, main_df[col], bottom=bottom_series, color=color, edgecolor=bar_edge_color, width=bar_thickness, label=col)
        if bottom_series is None:
            bottom_series = main_df[col]
        else:
            bottom_series += main_df[col]

    axes[0].tick_params(axis='x', labelsize=tick_fontsize)
    axes[0].tick_params(axis='y', labelsize=tick_fontsize)
    for tick in axes[0].get_xticklabels():
        tick.set_fontname(tick_fontname)
    for tick in axes[0].get_yticklabels():
        tick.set_fontname(tick_fontname)
    if y_range:
        axes[0].set_ylim(y_range[0])
    if is_legend:
        if cat_labels == None:
            axes[0].legend(fontsize=legend_size)
        else:
            axes[0].legend(cat_labels, fontsize=legend_size)

    # Plot each additional dataset as a line plot
    for idx, series in enumerate(all_series, start=1):
        
        axes[idx].tick_params(axis='x', labelsize=tick_fontsize)
        axes[idx].tick_params(axis='y', labelsize=tick_fontsize)
        for tick in axes[idx].get_xticklabels():
            tick.set_fontname(tick_fontname)
        for tick in axes[idx].get_yticklabels():
            tick.set_fontname(tick_fontname)
        if y_range:
            axes[idx].set_ylim(y_range[idx])
        if line_thickness is not None:
            if is_number(line_thickness):
                axes[idx].plot(series.index, series.values, color=line_color, linewidth=line_thickness)
            else:
                if line_thickness[idx - 1] != 0:
                    axes[idx].plot(series.index, series.values, color=line_color, linewidth=line_thickness[idx - 1])
                else:
                    axes[idx].scatter(series.index, series.values, color=line_color, s=1)
        else:
            axes[idx].plot(series.index, series.values, color=line_color)
    if not xlabels:
        xlabels = [''] * (1 + len(other_data_list))
    if not ylabels:
        ylabels = [''] * (1 + len(other_data_list))
    for i, ax in enumerate(axes):
        ax.set_xlabel(xlabels[i])
        ax.set_ylabel(ylabels[i])
        
    # Set date format on x-axis and set tick interval for all subplots
    axes[-1].xaxis.set_major_locator(mdates.DayLocator(interval=x_tick_interval))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

    # Apply rotation and alignment to x-tick labels
    ha_value = 'center' if rotation == 0 else 'right'
    plt.setp(axes[-1].get_xticklabels(), rotation=rotation, ha=ha_value)
    fig.subplots_adjust(hspace=0.1)
    #fig.tight_layout()
    if is_show:
        plt.show()
    if is_save:
        if save_path:
            fig.savefig(save_path, dpi=600)
        else:
            print("Please provide a valid path to save the figure.")
            
def plot_histograms(list_of_lists, titles, xlabels='', ylabels='', bins=10, color='blue', edgecolor='black', fig_size=(10, 6), tick_fontname='Arial',
                    tick_fontsize=12, title_fontsize=14, label_fontsize=14, value_range=None, line_color='red', show_all_xticklabels=True,
                    line_style='--', line_width=2, is_legend=False, unit='m', is_log=False, is_log_x=False, is_fixed_y_range=False, 
                    y_range=None, is_mean_value = True, label_sep='_', x_range = None,
                    is_scaled=False, scale_factor=10, save_path='', is_show=False, is_save=True, transparent_bg=True, hspace=0.1):
    
    '''
    This function is used to plot multiple histograms in one figure. The input data should be a list of lists, where each list contains the data for one histogram.
    
    Parameters:
    list_of_lists: list of lists, the input data for the histograms
    titles: list of str, the title of each histogram
    xlabels: str or list of str, the x label of each histogram, if it is empty, no x label will be shown
    ylabels: str or list of str, the y label of each histogram, if it is empty, no y label will be shown
    bins: int, the number of bins
    color: str, the color of the bars
    edgecolor: str, the edge color of the bars
    fig_size: tuple, the size of the figure
    tick_fontname: str, the font name of the ticks
    tick_fontsize: int, the font size of the ticks
    title_fontsize: int, the font size of the titles
    label_fontsize: int, the font size of the labels
    value_range: list, the range of the values
    line_color: str, the color of the mean line
    show_all_xticklabels: bool, whether to show all x tick labels
    line_style: str, the style of the mean line
    line_width: float, the width of the mean line
    is_legend: bool, whether to show the legend
    unit: str, the unit of the data
    is_log: bool, whether to plot the histogram in log scale
    is_log_x: bool, whether to plot the histogram in log scale for x axis
    is_fixed_y_range: bool, whether to fix the y range
    y_range: list, the y range
    is_mean_value: bool, whether to show the mean value
    is_scaled: bool, whether to scale the data
    scale_factor: float, the scale factor
    save_path: str, the path to save the figure
    is_show: bool, whether to show the figure
    is_save: bool, whether to save the figure
    transparent_bg: bool, whether to set the background to be transparent
    hspace: float, the space between subplots
    label_sep: str, the separator for the labels
    Returns:
    None
    
    If you want to customize the plot, you can modify the code in this function.
    
    '''
    
    n = len(list_of_lists)
    w, h = fig_size
    
    if is_fixed_y_range and y_range is None:
        max_frequency = 0
        for data in list_of_lists:
            if is_scaled:
                data = np.array(data) / scale_factor
            if is_log_x:
                min_data, max_data = min(data), max(data)
                bins = np.logspace(np.log10(min_data), np.log10(max_data), bins)
            hist, _ = np.histogram(data, bins=bins, range=value_range)
            max_frequency = max(max_frequency, hist.max())
        y_range = [0, max_frequency * 1.05]
    
    fig, axs = plt.subplots(n, 1, figsize=(w, h * n))
    if xlabels == None:
        xlabels = ['X'] * n
    elif isinstance(xlabels, str):
        xlabels = [xlabels] * n
    if ylabels == None:
        ylabels = ['Frequency (%)'] * n
    elif isinstance(ylabels, str):
        ylabels = [ylabels] * n
    for i, data in enumerate(list_of_lists):
        if is_scaled:
            data = np.array(data) / scale_factor
        
        if is_log_x:
            min_data, max_data = min(data), max(data)
            bins = np.logspace(np.log10(min_data), np.log10(max_data), bins)
            axs[i].hist(data, bins=bins, color=color, edgecolor=edgecolor, weights=np.ones_like(data) / len(data) * 100)
            axs[i].set_xscale('log')
        else:
            
            axs[i].hist(data, bins=bins, color=color, edgecolor=edgecolor, weights=np.ones_like(data) / len(data) * 100, range=value_range, log=is_log)
        
        # Calculate and plot the mean line
        if is_mean_value:
            mean_value = np.mean(data)
            axs[i].axvline(mean_value, color=line_color, linestyle=line_style, linewidth=line_width, label=f'Mean: {mean_value:.2f} {unit}')
        
        if titles[i]:
            axs[i].set_title(__process_text_labels(titles[i],sep=label_sep), fontsize=title_fontsize, fontname=tick_fontname)

        if xlabels[i]:
            axs[i].set_xlabel(__process_text_labels(xlabels[i], sep=label_sep), fontsize=label_fontsize, fontname=tick_fontname)
        else:
            axs[i].set_xticks([])
        
        if ylabels[i]:
            axs[i].set_ylabel(__process_text_labels(ylabels[i], sep=label_sep), fontsize=label_fontsize, fontname=tick_fontname)
        
        if is_legend:
            axs[i].legend(loc="upper left")
        
        axs[i].grid(False)
        axs[i].tick_params(axis='both', which='major', labelsize=tick_fontsize)
        
        if not show_all_xticklabels and i != n - 1:
            axs[i].set_xticklabels([])
        if is_fixed_y_range:
            axs[i].set_ylim(y_range)
        if transparent_bg:
            axs[i].patch.set_alpha(0)
        if x_range != None:
            axs[i].set_xlim(x_range)
    plt.tight_layout()
    plt.subplots_adjust(hspace=hspace)
    if is_show:
        plt.show()
    if is_save:
        if save_path:
            fig.savefig(save_path, dpi=600, transparent=transparent_bg)
        else:
            print("Please provide a valid path to save the figure.")
            
def plot_polylines(df, x, ys, line_styles=None, line_widths=None, line_colors=None, legends=None, show_legend=True,
                   marker_colors=None, fig_size=(10, 6), x_tick_interval=1, markers=None, y_label = None, label_sep = '_',
                   show_grid=True, font_name='Arial', font_size=12, save_path=None, dpi=600, y_range = None, is_show = True):
    """
    Plots multiple lines from a DataFrame using column indices for x and ys with customizable font settings
    and an option to save the figure.

    Parameters:
    df (DataFrame): The DataFrame containing the data.
    x (int): Index of the column to use as x-axis.
    ys (list of int): List of indices of columns to plot on the y-axis.
    line_styles (dict): Dictionary mapping column indices to line styles.
    line_widths (dict): Dictionary mapping column indices to line widths.
    line_colors (dict): Dictionary mapping column indices to line colors.
    legends (list): Optional list of legend labels.
    marker_colors (dict): Dictionary mapping column indices to marker colors.
    fig_size (tuple): Figure size.
    x_tick_interval (int): Interval between x-ticks.
    markers (dict): Dictionary mapping column indices to markers.
    show_grid (bool): Whether to show grid.
    font_name (str): Font name for all text elements.
    font_size (int): Font size for all text elements.
    save_path (str): Path to save the figure. If None, the figure is not saved.
    dpi (int): The resolution in dots per inch of the saved figure.
    label_sep (str): Separator for the labels.
    is_show (bool): Whether to display the plot.
    Returns:
    None
    
    """
    plt.figure(figsize=fig_size)
    ax = plt.gca()

    # Set global font properties
    plt.rcParams.update({'font.size': font_size, 'font.family': font_name})

    for y in ys:
        if line_styles is None:
            line_styles = {'-999':'-'}
        if line_widths is None:
            line_widths = {'-999':2}
        if line_colors is None:
            line_colors = {'-999':'blue'}
        if marker_colors is None:
            marker_colors = {'-999':'blue'}
        if markers is None:
            markers = {'-999':''}
        plt.plot(df.iloc[:, x], df.iloc[:, y],
                 linestyle=line_styles.get(y, '-'),  # Default to solid line
                 linewidth=line_widths.get(y, 2),    # Default line width
                 color=line_colors.get(y, 'blue'),   # Default line color
                 marker=markers.get(y, ''),          # Default no markers
                 markerfacecolor=marker_colors.get(y, 'blue'))  # Default marker color

    # Set x-ticks interval
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    # only show the ticks of intervals
    plt.xticks(np.arange(min(df.iloc[:, x]), max(df.iloc[:, x]) + 1, x_tick_interval))
    plt.xticks(rotation=0)
    plt.xlabel(__process_text_labels(df.columns[x], sep=label_sep), fontsize = font_size)
    y_label = "Percent (%)" if y_label is None else y_label
    plt.ylabel(__process_text_labels(y_label, sep = label_sep), fontsize = font_size)
    plt.title("")
    if show_grid:
        plt.grid(True)

    # Legend using column names or provided custom legends
    legend_labels = [df.columns[y] for y in ys] if not legends else legends
    if show_legend:
        plt.legend(legend_labels)
    
    if y_range:
        plt.ylim(y_range)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi)
        print(f"Figure saved to {save_path} at {dpi} dpi.")
    if is_show:
        plt.show()
    
def plot_time_histogram(histogram, color='blue', edgecolor='black', fig_size=(10, 6),
                        tick_fontname='Arial', tick_fontsize=12, title_fontsize=14, 
                        label_fontsize=14, y_range=None, 
                        x_tick_interval_minutes=5, x_ticklabel_format='HH:MM', 
                        is_legend=False, save_path='', is_show=True, 
                        is_save=False, transparent_bg=True):
    '''
    This function is used to plot a histogram of time ranges.
    
    Parameters:
    histogram: dict, the histogram of time ranges
    color: str, the color of the bars
    edgecolor: str, the edge color of the bars
    fig_size: tuple, the size of the figure
    tick_fontname: str, the font name of the ticks
    tick_fontsize: int, the font size of the ticks
    title_fontsize: int, the font size of the titles
    label_fontsize: int, the font size of the labels
    y_range: list, the y range
    x_tick_interval_minutes: int, the interval of the x ticks in minutes
    x_ticklabel_format: str, the format of the x tick labels
    is_legend: bool, whether to show the legend
    save_path: str, the path to save the figure
    is_show: bool, whether to show the figure
    is_save: bool, whether to save the figure
    transparent_bg: bool, whether to set the background to be transparent
    
    '''
    
    fig, ax = plt.subplots(figsize=fig_size)
    
    # Convert the time range strings to datetime objects and calculate bar widths
    starts = []
    ends = []
    values = list(histogram.values())
    for time_range in histogram.keys():
        start_time_str, end_time_str = time_range.split(' - ')
        start_datetime = datetime.strptime(start_time_str, '%H:%M:%S')
        end_datetime = datetime.strptime(end_time_str, '%H:%M:%S')
        starts.append(start_datetime)
        ends.append(end_datetime)
    
    # Plotting the bar chart
    for start, end, value in zip(starts, ends, values):
        width = end - start  # timedelta object
        # Convert width from timedelta to the fraction of the day for plotting
        width_in_days = width.total_seconds() / 86400
        ax.bar(start + width/2, value, width=width_in_days, color=color, edgecolor=edgecolor, align='center')
    
    if y_range:
        ax.set_ylim(y_range)

    ax.xaxis.set_major_formatter(mdates.DateFormatter(x_ticklabel_format))
    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=x_tick_interval_minutes))
    
    ax.tick_params(axis='both', labelsize=tick_fontsize, labelcolor='black')
    plt.xticks(fontname=tick_fontname)
    plt.yticks(fontname=tick_fontname)
    
    ax.set_title('Histogram of Time Ranges', fontsize=title_fontsize)
    ax.set_xlabel('Time', fontsize=label_fontsize)
    ax.set_ylabel('Frequency', fontsize=label_fontsize)
    
    if is_legend:
        ax.legend()

    if is_save and save_path:
        plt.savefig(save_path, transparent=transparent_bg, bbox_inches='tight', dpi=600)
    
    if is_show:
        plt.show()
    
    plt.close()

def plot_2D_heatmap(pair_freq, x_bin_ticks=None, y_bin_ticks=None, fig_size=(10, 8), title='Cross Relationship Heatmap', title_fontsize=16,
                 xlabel='Variable 1', ylabel='Variable 2', label_fontsize=14, tick_fontsize=12, vmin = None, vmax = None,
                 cmap='viridis', cbar_label='Frequency', save_path='', is_show=True, x_ticklabel_left_close = False, x_ticklabel_right_close = False,
                 y_ticklabel_top_close = False, y_ticklabel_bottom_close = False,is_annotate = False, is_percent = True, label_sep = '_',
                 xtick_rotation=0, ytick_rotation=0, xticklabels=None, yticklabels=None):
    """
    Plots a heatmap of the frequency of tuple pairs.
    
    :param pair_freq: Dictionary with keys as tuple pairs of values and values as their frequency, or orig data with two columns.
    :param fig_size: Size of the figure.
    :param title: Title of the heatmap.
    :param title_fontsize: Font size for the title.
    :param xlabel: Label for the x-axis.
    :param ylabel: Label for the y-axis.
    :param label_fontsize: Font size for labels.
    :param tick_fontsize: Font size for tick labels.
    :param cmap: Color map of the heatmap.
    :param vmin: Minimum value for the color bar.
    :param vmax: Maximum value for the color bar.
    :param is_percent: Whether to show the frequency as a percentage.
    :param cbar_label: Label for the color bar.
    :param save_path: Path to save the figure.
    :param is_show: Whether to display the plot.
    :param x_ticklabel_left_close: Whether to close the left side of x-tick labels.
    :param x_ticklabel_right_close: Whether to close the right side of x-tick labels.
    :param y_ticklabel_top_close: Whether to close the top side of y-tick labels.
    :param y_ticklabel_bottom_close: Whether to close the bottom side of y-tick labels.
    :param is_annotate: Whether to annotate the heatmap with the frequency values.
    :param label_sep: Separator for the labels.
    :param xtick_rotation: Rotation angle of x-tick labels.
    :param ytick_rotation: Rotation angle of y-tick labels.
    :param xticklabels: Custom labels for the x-axis ticks.
    :param yticklabels: Custom labels for the y-axis ticks.
    """
    if isinstance(pair_freq, pd.DataFrame):
        if x_bin_ticks is None:
            x_vals = pair_freq.iloc[:, 0].values
            x_bin_nums = 2 if max(x_vals) - min(x_vals) <= 1 else min(10, max(x_vals) - min(x_vals))
            x_bin_ticks = __generate_bin_ticks(x_vals, x_bin_nums, mode='range')
        if y_bin_ticks is None:
            y_vals = pair_freq.iloc[:, 0].values
            y_bin_nums = 2 if max(y_vals) - min(y_vals) <= 1 else min(10, max(y_vals) - min(y_vals))
            y_bin_ticks = __generate_bin_ticks(y_vals, y_bin_nums, mode='range')
        key1 = pair_freq.columns[0]
        key2 = pair_freq.columns[1]
        
        pair_frequency = __analyze_cross_relationship(pair_freq, key1, key2, x_bin_ticks, y_bin_ticks)
    else:
        pair_frequency = copy.deepcopy(pair_freq)
    
    index = list(range(0, len(x_bin_ticks) - 1))
    columns = list(range(0, len(y_bin_ticks) - 1))
    
    if not x_ticklabel_left_close:
        columns.append(columns[-1] + 1)
    if not x_ticklabel_right_close:
        columns.append(columns[-1] + 1)
    if not y_ticklabel_top_close:
        index.append(index[-1] + 1)
    if not y_ticklabel_bottom_close:
        index.append(index[-1] + 1)

    # Create a DataFrame from the pair_frequency
    data = np.zeros((len(columns), len(index)))
    for (var1, var2), freq in pair_frequency.items():
        i = index.index(var1)
        j = columns.index(var2)
        data[j, i] = freq
    
    if is_percent:
        data = data / np.sum(data) * 100
        
    df = pd.DataFrame(data, index=columns, columns=index)

    # Plotting
    plt.figure(figsize=fig_size)
    if vmin is None or vmax is None:
        heatmap = sns.heatmap(df, annot=is_annotate, fmt=".0f", cmap=cmap, linewidths=.5, 
                          cbar_kws={'label': cbar_label})
    else:
        heatmap = sns.heatmap(df, annot=is_annotate, fmt=".0f", cmap=cmap, linewidths=.5, vmin=vmin, vmax=vmax,
                          cbar_kws={'label': cbar_label})
    plt.title(__process_text_labels(title, sep=label_sep), fontsize=title_fontsize)
    plt.xlabel(__process_text_labels(xlabel, sep=label_sep), fontsize=label_fontsize)
    plt.ylabel(__process_text_labels(ylabel, sep=label_sep), fontsize=label_fontsize)

    # Custom or default tick labels
    if xticklabels is None:
        xticklabels = __generate_ticklabels(x_bin_ticks, x_ticklabel_left_close, x_ticklabel_right_close)
        
    plt.xticks(ticks=np.arange(len(index)) + 0.5, labels=xticklabels, rotation=xtick_rotation, fontsize=tick_fontsize)
    if yticklabels is None:
        yticklabels = __generate_ticklabels(y_bin_ticks, y_ticklabel_top_close, y_ticklabel_bottom_close)
        
    plt.yticks(ticks=np.arange(len(columns)) + 0.5, labels=yticklabels, rotation=ytick_rotation, fontsize=tick_fontsize)
    # Save the plot
    if save_path:
        plt.savefig(save_path, bbox_inches='tight',dpi=600)
    
    # Show the plot
    if is_show:
        plt.show()
    plt.close()

def plot_contour_map(data, fig_size=(10, 8), title='Contour Map', title_fontsize=16,
                     xlabel='X Coordinate', ylabel='Y Coordinate', colorbar_label='Frequency (%)', label_fontsize=14,
                     tick_fontsize=12, contour_levels=20, cmap='viridis', is_contour_line=True,
                     contour_line_colors='white', contour_line_width=1.0, is_contour_label=True,
                     contour_label_color='white', contour_label_size=10, label_interval=2,
                     save_path='', is_show=True, xticks=None, yticks=None, xtick_labels=None, ytick_labels=None):
    """
    Plots a contour map for data with optional tick intervals and custom labels.
    
    :param data: Dictionary or DataFrame containing the data to plot.
    :param fig_size: Size of the figure.
    :param title: Title of the plot.
    :param title_fontsize: Font size for the title.
    :param xlabel: Label for the x-axis.
    :param ylabel: Label for the y-axis.
    :param colorbar_label: Label for the colorbar.
    :param label_fontsize: Font size for labels.
    :param tick_fontsize: Font size for tick labels.
    :param contour_levels: Number of contour levels.
    :param cmap: Colormap for the plot.
    :param is_contour_line: Whether to display contour lines.
    :param contour_line_colors: Color of the contour lines.
    :param contour_line_width: Width of the contour lines.
    :param is_contour_label: Whether to display contour labels.
    :param contour_label_color: Color of the contour labels.
    :param contour_label_size: Font size of the contour labels.
    :param label_interval: Interval for displaying contour labels.
    :param save_path: Path to save the figure.
    :param is_show: Whether to display the plot.
    :param xticks: Custom tick intervals for the x-axis.
    :param yticks: Custom tick intervals for the y-axis.
    :param xtick_labels: Custom labels for the x-axis ticks.
    :param ytick_labels: Custom labels for the y-axis ticks.
    
    """
    if isinstance(data, dict):
        points = np.array([(key[0], key[1]) for key in data.keys()])
        values = np.array(list(data.values()))
    elif isinstance(data, pd.DataFrame):
        if data.shape[1] != 2:
            raise ValueError("DataFrame must have exactly two columns for x and y values.")
        points = np.array([(key[0], key[1]) for key in data.iloc[:, 0].values])
        values = data.iloc[:, -1].values
        print(points)
        print(values)
    else:
        raise TypeError("Input data must be a dictionary or a pandas DataFrame.")

    # Generate a grid to interpolate onto
    x = np.linspace(np.min(points[:, 0]), np.max(points[:, 0]), 100)
    y = np.linspace(np.min(points[:, 1]), np.max(points[:, 1]), 100)
    X, Y = np.meshgrid(x, y)
    
    # Interpolate data
    Z = griddata(points, values, (X, Y), method='cubic')

    # Plotting
    plt.figure(figsize=fig_size)
    plt.contourf(X, Y, Z, levels=contour_levels, cmap=cmap)
    cbar = plt.colorbar()
    cbar.set_label(colorbar_label, size=label_fontsize)
    cbar.ax.tick_params(labelsize=tick_fontsize)

    if is_contour_line:
        contour_lines = plt.contour(X, Y, Z, levels=contour_levels, colors=contour_line_colors, linewidths=contour_line_width)
    if is_contour_line and is_contour_label:
        labels = plt.clabel(contour_lines, inline=True, fontsize=contour_label_size, colors=contour_label_color)
        for label in labels:
            label.set_visible(False)
        for i, label in enumerate(labels):
            if i % label_interval == 0:
                label.set_visible(True)

    plt.title(title, fontsize=title_fontsize)
    plt.xlabel(xlabel, fontsize=label_fontsize)
    plt.ylabel(ylabel, fontsize=label_fontsize)

    # Set custom tick intervals and labels if provided
    if xticks is not None:
        plt.xticks(xticks, xtick_labels if xtick_labels is not None else xticks, fontsize=tick_fontsize)
    else:
        plt.xticks(fontsize=tick_fontsize)
    if yticks is not None:
        plt.yticks(yticks, ytick_labels if ytick_labels is not None else yticks, fontsize=tick_fontsize)
    else:
        plt.yticks(fontsize=tick_fontsize)

    # Save the plot
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    # Show the plot
    if is_show:
        plt.show()
    plt.close()
    
def plot_3d_stacked_bar(pair_freq, x_bin_ticks, y_bin_ticks, fig_size=(12, 10), title='3D Stacked Bar Plot', title_fontsize=16,
                        xlabel='Variable 1', ylabel='Variable 2', zlabel='Frequency', label_fontsize=10, tick_fontsize=10, grid_alpha=0.5,
                        color_map=plt.cm.viridis, save_path='', is_show=True, x_ticklabel_left_close = False, x_ticklabel_right_close = False,
                        y_ticklabel_top_close = False, y_ticklabel_bottom_close = False, zmin = None, zmax = None, grid_style='dashed',
                        xtick_rotation=0, ytick_rotation=0, xticklabels=None, yticklabels=None, is_percent=False, elevation=30, azimuth=30,
                        background_color='white'):
    """
    Plots a 3D stacked bar plot of the frequency of tuple pairs.

    :param pair_freq: Dictionary with keys as tuple pairs of values (x, y) and values as their frequency.
    :param fig_size: Size of the figure.
    :param title: Title of the plot.
    :param title_fontsize: Font size for the title.
    :param xlabel: Label for the x-axis.
    :param ylabel: Label for the y-axis.
    :param zlabel: Label for the z-axis.
    :param is_percent: Whether to display the z-axis as a percentage.
    :param label_fontsize: Font size for labels.
    :param tick_fontsize: Font size for tick labels.
    :param color_map: Color map for the bars.
    :param save_path: Path to save the figure.
    :param is_show: Whether to display the plot.
    :param x_ticklabel_left_close: Whether to close the left side of x-tick labels.
    :param x_ticklabel_right_close: Whether to close the right side of x-tick labels.
    :param y_ticklabel_top_close: Whether to close the top side of y-tick labels.
    :param y_ticklabel_bottom_close: Whether to close the bottom side of y-tick labels.
    :param xtick_rotation: Rotation angle of x-tick labels.
    :param ytick_rotation: Rotation angle of y-tick labels.
    :param xticklabels: Custom labels for the x-axis ticks.
    :param yticklabels: Custom labels for the y-axis ticks.
    :param grid_alpha: Transparency of the grid lines.
    :param zmin: Minimum value for the z-axis.
    :param zmax: Maximum value for the z-axis.
    :param elevation: Elevation angle of the plot.
    :param azimuth: Azimuth angle of the plot.
    :param grid_style: Style of the grid lines, solid or dashed.
    :param background_color: Background color of the plot.
    """
    if isinstance(pair_freq, pd.DataFrame):
        if x_bin_ticks is None:
            x_bin_ticks = __generate_bin_ticks(pair_freq.iloc[:, 0], 10, mode='range')
        if y_bin_ticks is None:
            y_bin_ticks = __generate_bin_ticks(pair_freq.iloc[:, 1], 10, mode='range')
        key1 = pair_freq.columns[0]
        key2 = pair_freq.columns[1]
        pair_frequency = __analyze_cross_relationship(pair_freq, key1, key2, x_bin_ticks, y_bin_ticks)
    else:
        pair_frequency = copy.deepcopy(pair_freq)
    
    pair_sum = sum(pair_frequency.values())
    if is_percent:
        pair_frequency = {key: val / pair_sum * 100 for key, val in pair_frequency.items()}
    
    fig = plt.figure(figsize=fig_size)
    fig.patch.set_facecolor(background_color)  # Set the figure background color
    ax = fig.add_subplot(111, projection='3d')
    
    xs = list(range(0, len(x_bin_ticks) - 1))
    ys = list(range(0, len(y_bin_ticks) - 1))
    
    if not x_ticklabel_left_close:
        xs.append(xs[-1] + 1)
    if not x_ticklabel_right_close:
        xs.append(xs[-1] + 1)
    if not y_ticklabel_top_close:
        ys.append(ys[-1] + 1)
    if not y_ticklabel_bottom_close:
        ys.append(ys[-1] + 1)
    
    # Create data arrays
    xs = np.arange(len(x_bin_ticks) + 1)
    ys = np.arange(len(y_bin_ticks) + 1)
    data = np.zeros((len(ys), len(xs)))

    for (x, y), freq in pair_frequency.items():
        # print(x, y, freq)
        x_index = xs.tolist().index(x)
        y_index = ys.tolist().index(y)
        data[y_index, x_index] += freq
    
    # print(data)
    # Create bars
    for x in xs:
        for y in ys:
            ax.bar3d(x, y, 0, 1, 1, data[y, x], color=color_map(data[y, x] / np.max(data)))

    ax.set_title(title, fontsize=title_fontsize)
    ax.set_xlabel(xlabel, fontsize=label_fontsize, labelpad = 10)
    ax.set_ylabel(ylabel, fontsize=label_fontsize)
    ax.set_zlabel(zlabel, fontsize=label_fontsize)

    if xticklabels is None:
        xticklabels = __generate_ticklabels(x_bin_ticks, x_ticklabel_left_close, x_ticklabel_right_close)
    
    if yticklabels is None:
        yticklabels = __generate_ticklabels(y_bin_ticks, y_ticklabel_top_close, y_ticklabel_bottom_close)
    # print(xs)
    # print(xticklabels)
    ax.set_xticks(xs + 0.5)
    ax.set_xticklabels(xticklabels, fontsize=tick_fontsize, rotation=xtick_rotation, ha='center')

    ax.set_yticks(ys + 0.5)
    ax.set_yticklabels(yticklabels, fontsize=tick_fontsize, rotation=ytick_rotation, ha = 'left', va='bottom')
    
    if zmin is not None and zmax is not None:
        ax.set_zlim3d(zmin, zmax)
    ax.set_zticklabels(ax.get_zticks(), fontsize=tick_fontsize)
    # Adjust the view angle
    ax.view_init(elev=elevation, azim=azimuth)  # Apply view angle settings
    
    # Make gridlines more transparent
    gridline_style = {'linestyle': grid_style, 'alpha': grid_alpha}
    ax.xaxis._axinfo["grid"].update(gridline_style)
    ax.yaxis._axinfo["grid"].update(gridline_style)
    ax.zaxis._axinfo["grid"].update(gridline_style)
    
    # Save the plot
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=600)
    
    # Show the plot
    if is_show:
        plt.show()
    plt.close()
    
def plot_rose_map(input_data, key_1, key_2, interval=10, value_interval=None,
                  title="Rose Map", label_size=12, label_interval=1, color_ramp="viridis", 
                  tick_label_size = 12, tick_label_color = 'black', tick_font_name='Arial', 
                  fig_size=(10, 8), colorbar_label='Intensity',
                  colorbar_label_size=12, max_radius = None, save_path='', is_show=True):
    """
    Plots a rose map for directional data with optional value binning.
    
    Args:
    input_data (pd.DataFrame): The input DataFrame containing the data.
    key_1 (str): The column name for the directional data.
    key_2 (str): The column name for the value data.
    interval (int): The bin size for the directional data in degrees.
    value_interval (int): The bin size for the value data.
    title (str): The title of the plot.
    label_size (int): The font size for the labels.
    label_interval (int): The interval for displaying labels on the plot.
    color_ramp (str): The color ramp to use for the plot.
    tick_label_size (int): The font size for the tick labels.
    tick_label_color (str): The color for the tick labels.
    tick_font_name (str): The font name for the tick labels.
    fig_size (tuple): The size of the figure.
    colorbar_label (str): The label for the colorbar.
    colorbar_label_size (int): The font size for the colorbar label.
    max_radius (float): The maximum radius for the plot.
    save_path (str): The path to save the plot.
    is_show (bool): Whether to display the plot.
    
    """
    # Check for required columns
    if key_1 not in input_data.columns or key_2 not in input_data.columns:
        raise ValueError(f"Columns {key_1} and {key_2} must exist in the DataFrame.")

    num_bins = int(360 / interval)
    bin_edges = np.linspace(0, 360, num_bins + 1)
    input_data['binned_direction'] = pd.cut(input_data[key_1], bins=bin_edges, labels=bin_edges[:-1], right=False)

    if value_interval:
        max_value = input_data[key_2].max()
        value_bin_edges = np.arange(0, max_value + value_interval, value_interval)
        input_data['binned_value'] = pd.cut(input_data[key_2], bins=value_bin_edges, labels=value_bin_edges[:-1], right=False)
        grouped = input_data.groupby(['binned_direction', 'binned_value'])[key_2].count().unstack(fill_value=0)
        stats = grouped
    else:
        grouped = input_data.groupby('binned_direction')[key_2].agg(['max', 'mean'])
        stats = grouped

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=fig_size)
    cmap = get_cmap(color_ramp)

    # Plot each segment if value_interval is used
    norm = Normalize(vmin=0, vmax=max_value if value_interval else stats['mean'].max())
    if value_interval:
        for idx, row in stats.iterrows():
            base_radius = 0
            for value_idx, count in enumerate(row):
                if count > 0:
                    theta = float(idx) * np.pi / 180
                    radius = count
                    color = cmap(norm(value_idx * value_interval + value_interval / 2))
                    ax.bar(theta, radius, width=np.deg2rad(interval), bottom=base_radius, color=color, edgecolor='black', align='edge')
                    base_radius += radius
    else:
        radii = stats['max']
        colors = [cmap(norm(value)) for value in stats['mean']]
        bars = ax.bar(stats.index.astype(float) * np.pi / 180, radii, width=np.deg2rad(interval), color=colors, edgecolor='black', align='edge')

    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_xticks(np.linspace(0, 2 * np.pi, num_bins, endpoint=False))
    ax.set_xticklabels([f"{int(angle)}°" if i % label_interval == 0 else '' for i, angle in enumerate(np.linspace(0, 360, num_bins, endpoint=False))], fontsize=label_size, fontname=tick_font_name)
    ax.set_title(title, va='bottom', fontsize=label_size + 2)

    ax.yaxis.set_tick_params(labelsize=tick_label_size, colors=tick_label_color)
    
    # Customize radial axis
    if max_radius is None:
        max_radius = stats['max'].max() if not value_interval else value_bin_edges[-1]
    ax.set_ylim(0, max_radius)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.1)
    cbar.set_label(colorbar_label, size=colorbar_label_size)
    cbar.ax.tick_params(labelsize=tick_label_size)
    if is_show:
        plt.show()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi = 600)
        
def plot_rose_contour_map(input_data, key_1, key_2, title="Rose Contour Map", label_size=12, color_ramp="viridis",
                          fig_size=(10, 8), num_levels=10, max_radius=None, density_threshold=None, z_label = 'Density',
                          boundary_line_color='black', boundary_line_thickness=2, is_percent = True, tick_spacing = 2,
                          save_path='', is_show=True):
    """
    Plots a rose contour map for directional data with a boundary line at the density threshold, handling cases where no contours are found.
    
    Args:
    input_data (pd.DataFrame): The input DataFrame containing the data.
    key_1 (str): The column name for the directional data.
    key_2 (str): The column name for the value data.
    title (str): The title of the plot.
    label_size (int): The font size for the labels.
    color_ramp (str): The color ramp to use for the plot.
    fig_size (tuple): The size of the figure.
    num_levels (int): The number of contour levels to plot.
    max_radius (float): The maximum radius for the plot.
    density_threshold (float): The density threshold for the boundary line.
    z_label (str): The label for the colorbar.
    boundary_line_color (str): The color for the boundary line.
    boundary_line_thickness (float): The thickness of the boundary line.
    is_percent (bool): Whether to show the colorbar as a percentage.
    tick_spacing (int): The spacing between ticks on the colorbar.
    save_path (str): The path to save the plot.
    is_show (bool): Whether to display the plot.
    
    """
    if key_1 not in input_data.columns or key_2 not in input_data.columns:
        raise ValueError(f"Columns {key_1} and {key_2} must exist in the DataFrame.")

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=fig_size)

    # Convert directions to radians for plotting
    theta = np.radians(input_data[key_1])
    r = input_data[key_2]

    # Set up the KDE for the polar data
    k = kde.gaussian_kde([theta, r])
    t_i = np.linspace(0, 2 * np.pi, 360)
    r_i = np.linspace(0, r.max() if max_radius is None else max_radius, 100)
    T, R = np.meshgrid(t_i, r_i)
    Z = k(np.vstack([T.ravel(), R.ravel()])).reshape(T.shape)
    # print(Z)
    # Normalize and create the colormap
    norm = Normalize(vmin=Z.min(), vmax=Z.max())
    
    original_cmap = get_cmap(color_ramp)

    colors = original_cmap(np.linspace(0, 1, original_cmap.N))
    
    if density_threshold is not None:
        # Check if the density threshold is within the KDE output range
        if not (Z.min() <= density_threshold <= Z.max()):
            print("Warning: Density threshold is out of the data range. Adjusting to fit within the range.")
            density_threshold = Z.min() + (Z.max() - Z.min()) * 0.1  # Example adjustment
        # Set the colors for the threshold and below to white
        ind = (density_threshold - Z.min()) / (Z.max() - Z.min()) * len(colors)
        
        colors[: int(ind)] = [1, 1, 1, 1]  # RGBA for white
        new_cmap = ListedColormap(colors)
    else:
        new_cmap = original_cmap

    color_levels = __generate_levels(Z, num_levels, density_threshold = density_threshold)
    # Plotting the contour map
    c = ax.contourf(T, R, Z, levels=color_levels, norm=norm, cmap=new_cmap, extend = 'min')

    # Optionally add a boundary line at the threshold
    if density_threshold is not None:
        try:
            cs = ax.contour(T, R, Z, levels=[density_threshold], colors=boundary_line_color, linewidths=boundary_line_thickness)
            for collection in cs.collections:
                for path in collection.get_paths():
                    ax.plot(path.vertices[:, 0], path.vertices[:, 1], color=boundary_line_color, linewidth=boundary_line_thickness)
        except ValueError:
            print("No valid contours were found at the specified threshold.")

    # Colorbar
    cbar = plt.colorbar(c, ax=ax, orientation='vertical')
    cbar.set_label(z_label, size=label_size)
    cbar.ax.tick_params(labelsize=label_size)
    
    tick_digits = __find_tick_digits(color_levels[0]) - 1
 
    tick_interval = 10 ** tick_digits * tick_spacing
    
    if density_threshold is not None:
        min_val = __round_to_nearest(density_threshold, tick_digits)
    else:
        min_val = __round_to_nearest(Z.min(), tick_digits)
    tick_positions = np.arange(min_val, __round_to_nearest(Z.max(), tick_digits), tick_interval)
    cbar.set_ticks(tick_positions)
    if is_percent:
        cbar.set_ticklabels([f'{__round_to_nearest(100 * i, tick_digits + 2)}' for i in tick_positions])
    else:
        cbar.set_ticklabels([f'{i}' for i in tick_positions])
    # print(Z.min(), Z.max(), tick_digits, 10**tick_digits)
    # print(__round_to_nearest(Z.max(), tick_digits - 1))
    # print(tick_positions)
    # Setting labels and title
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_xticks(np.linspace(0, 2 * np.pi, 8, endpoint=False))
    ax.set_xticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])
    ax.set_title(title, va='bottom', fontsize=label_size + 2)
    if is_show:
        plt.show()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=600)

def plot_cdfs(data_lists, fig_size=(10, 6), line_styles=None, line_widths=None, is_percent=True,
              line_colors=None, legends=None, marker_colors=None, x_tick_interval=10, tick_label_fontsize=12,
              markers=None, show_grid=True, font_name='Arial', font_size=12, save_path=None,
              dpi=600, is_same_figure=True, is_log_x=False, regression_trans=None, title='Cumulative Distribution Function',
              show_reg_eqn=True, annno_digits=3, x_lim=None, x_label=None, y_label=None,
              labelpad=5, titlepad=10):
    """
    Plot the CDF for each set of data in data_lists with extensive customization options,
    including the option to plot all on the same figure or on individual subplots, and
    setting the x-axis to logarithmic scale. Additionally, performs OLS regression
    within specified ranges and plots the regression lines.
    """
    if is_same_figure:
        fig, ax = plt.subplots(figsize=fig_size)
    else:
        fig, axs = plt.subplots(len(data_lists), 1, figsize=fig_size)
    
    plt.rc('font', family=font_name, size=font_size)
    
    for i, data in enumerate(data_lists):
        sorted_data = np.sort(data)
        yvals = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        
        if is_percent:
            yvals = yvals * 100
        style = line_styles.get(i, '-') if line_styles else '-'
        width = line_widths.get(i, 1) if line_widths else 1
        color = line_colors.get(i, 'b') if line_colors else 'b'
        marker = markers.get(i, None) if markers else None
        marker_color = marker_colors.get(i, color) if marker_colors else color
        legend = legends[i] if legends and i < len(legends) else f"Data {i+1}"

        current_ax = ax if is_same_figure else axs[i]
        
        if is_log_x:
            current_ax.set_xscale('log')
            current_ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
            current_ax.xaxis.set_minor_formatter(mticker.NullFormatter())
            current_ax.ticklabel_format(style='plain', axis='x')

        current_ax.plot(sorted_data, yvals, linestyle=style, linewidth=width, color=color,
                        marker=marker, markerfacecolor=marker_color, label=legend)
        
        # Regression analysis within specified ranges
        if regression_trans:
            for start, end in regression_trans:
                mask = (sorted_data >= start) & (sorted_data <= end)
                reg_data = sorted_data[mask]
                reg_yvals = yvals[mask]

                if len(reg_data) > 1:
                    reg_model = sm.OLS(reg_yvals, sm.add_constant(reg_data)).fit()
                    reg_line = reg_model.predict(sm.add_constant(reg_data))
                    current_ax.plot(reg_data, reg_line, linestyle='--', linewidth=width, color=color)
                    if show_reg_eqn:
                        b, m = reg_model.params
                        equation_text = f"$y = {m:.{annno_digits}f}x + {b:.{annno_digits}f}$"
                        x_pos = np.median(reg_data)
                        y_pos = np.interp(x_pos, reg_data, reg_line)
                        current_ax.annotate(equation_text, (x_pos, y_pos), textcoords="offset points",
                                            xytext=(0,10), ha='right', color=color, fontsize=font_size)
        
        if x_tick_interval and not is_log_x:
            current_ax.set_xticks(np.arange(0, 1.1, 1 / x_tick_interval))
        
        if x_lim:
            current_ax.set_xlim(x_lim)

        if show_grid:
            current_ax.grid(True)
        current_ax.tick_params(axis='both', which='major', labelsize=tick_label_fontsize)
        current_ax.legend()

        if x_label:
            current_ax.set_xlabel(x_label, fontsize=font_size, labelpad=labelpad)
        if y_label:
            current_ax.set_ylabel(y_label, fontsize=font_size, labelpad=labelpad)

    plt.title(title, pad=titlepad)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_path:
        plt.savefig(save_path, dpi=dpi)
    
    plt.show()

def plot_intensity_velocity_and_classes(intensity_data, velocity_list, classification_data, v_choice='component',
                                                 primary_colormap='jet', edge_colormap='rainbow',
                                                 edge_thickness=2, face_alpha=1.0, arrow_scale=0.1,
                                                 face_label = None, edge_label = None, class_labels = None, is_legend = True,
                                                 is_show = True, save_path = None,
                                                 arrow_colors='black', arrow_styles='->', arrow_thicknesses=1, fig_size=(10, 8)):
    """
    Plots a heatmap with edges, velocity arrows, and classification patterns.

    Parameters:
    - intensity_data (list of np.array): List containing up to two 2D numpy arrays with intensity values.
    - velocity_list (list of np.array): List containing pairs of arrays representing velocity components.
    - classification_data (list of np.array): List containing a single array with classification integers for each cell.
    - v_choice (str): 'component' for Cartesian coordinates, 'theta' for polar coordinates (magnitude, direction).
    - primary_colormap (str), edge_colormap (str): Colormap identifiers.
    - edge_thickness (float): Thickness of the edges.
    - face_alpha (float): Transparency of the cell faces.
    - arrow_scale (float): Scaling factor for the arrows.
    - face_label (str), edge_label (str): Labels for the colorbars.
    - class_labels (list): List of class labels for the legend.
    - is_legend (bool): Whether to display the legend.
    - is_show (bool): Whether to display the plot.
    - save_path (str): Path to save the plot.
    - arrow_colors, arrow_styles, arrow_thicknesses: Properties for arrows.
    - fig_size (tuple): Figure size.
    """
    fig, ax = plt.subplots(figsize=fig_size)
    data = intensity_data[0]
    n_rows, n_cols = data.shape

    # Normalize the face colors
    norm = Normalize(vmin=np.min(data), vmax=np.max(data))
    face_cmap = get_cmap(primary_colormap)
    face_colors = face_cmap(norm(data))

    # Prepare patterns based on classification data
    patterns = ['/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*', '///', '\\\\', '|||', '---', 'xxx', 'ooo', 'OOO', '...', '***']
    class_data = classification_data[0]
    if class_labels ==  None:
        pattern_handles = [patches.Patch(facecolor='white', edgecolor = 'black', hatch=pat, label=f'Class {i+1}') \
            for i, pat in enumerate(patterns[:len(np.unique(class_data))])]
    else:
        pattern_handles = [patches.Patch(facecolor='white', edgecolor = 'black', hatch=pat, label=class_labels[i]) \
            for i, pat in enumerate(patterns[:len(np.unique(class_data))])]
    # Prepare edge colors if provided
    if len(intensity_data) > 1:
        edge_data = intensity_data[1]
        edge_norm = Normalize(vmin=np.min(edge_data), vmax=np.max(edge_data))
        edge_cmap = get_cmap(edge_colormap)
        edge_colors = edge_cmap(edge_norm(edge_data))
    else:
        edge_data = None
        
    for i in range(n_rows):
        for j in range(n_cols):
            face_color = face_colors[i, j]
            pattern = patterns[class_data[i, j] % len(patterns)]
            if len(intensity_data) > 1:
                edge_color = edge_colors[i, j]
                # Two rectangles to manage separate face and edge coloring
                rect0 = plt.Rectangle((j + 0.01, n_rows - i - 1 + 0.01), 0.99, 0.99, facecolor=face_color,
                                      alpha=face_alpha)
                ax.add_patch(rect0)
                
                pattern_rect = patches.Rectangle((j, n_rows - i - 1), 1, 1, hatch=pattern, fill=False,
                                                edgecolor='black', linewidth=0)
                ax.add_patch(pattern_rect)
                
                rect1 = plt.Rectangle((j + 0.05, n_rows - i - 1 + 0.05), 0.9, 0.9, facecolor='none',
                                      edgecolor=edge_color, linewidth=edge_thickness)
                ax.add_patch(rect1)
                
            else:
                rect = plt.Rectangle((j + 0.01, n_rows - i - 1 + 0.01), 0.99, 0.99, facecolor=face_color,
                                     edgecolor='black', linewidth=edge_thickness, alpha=face_alpha)
                ax.add_patch(rect)
                pattern_rect = patches.Rectangle((j, n_rows - i - 1), 1, 1, hatch=pattern, fill=False,
                                                edgecolor='white' if np.mean(face_color) < 0.5 else 'black', linewidth=0)
                ax.add_patch(pattern_rect)
                
                # Add pattern with transparency
                # Add pattern with contrasting color
            

            # Center of the cell for placing the arrow
            x_center, y_center = j + 0.5, n_rows - i - 0.5

            # Process each pair of arrays in velocity_list
            for k in range(0, len(velocity_list), 2):
                if v_choice == 'component':
                    dx, dy = velocity_list[k][i, j], velocity_list[k+1][i, j]
                elif v_choice == 'theta':
                    magnitude = velocity_list[k][i, j]
                    direction = velocity_list[k+1][i, j]
                    dx = magnitude * np.cos(direction)
                    dy = magnitude * np.sin(direction)

                # Get arrow properties
                color = arrow_colors[k // 2] if isinstance(arrow_colors, list) else arrow_colors
                style = arrow_styles[k // 2] if isinstance(arrow_styles, list) else arrow_styles
                thickness = arrow_thicknesses[k // 2] if isinstance(arrow_thicknesses, list) else arrow_thicknesses

                ax.arrow(x_center, y_center, dx * arrow_scale, dy * arrow_scale, color=color,
                         head_width=0.1, head_length=0.15, linewidth=thickness, linestyle=style)
    if is_legend:
        ax.legend(handles=pattern_handles, title="Classification", handler_map={patches.Patch: __HandlerRect()}, loc='upper left', bbox_to_anchor=(0.98, 0.98))
    # Colorbar for intensity
     # Adding colorbar for face colors
    
    ax.axis('off')  # Turn off axis markers, labels, and grid
    
    if is_legend:
        face_label = face_label if face_label is not None else 'Intensity'
        edge_label = edge_label if edge_label is not None else 'Edge Intensity'
        
        # Adding colorbar for face colors
        face_cbar_ax = fig.add_axes([0.9, 0.15, 0.015, 0.4])  # Adjust these values as needed
        sm_face = plt.cm.ScalarMappable(cmap=__create_transparent_cmap(face_cmap, alpha=face_alpha), norm=norm)
        sm_face.set_array([])
        cbar_face = plt.colorbar(sm_face, cax=face_cbar_ax, orientation='vertical')
        cbar_face.set_label(face_label, fontsize=10)  # Smaller label size
        cbar_face.ax.tick_params(labelsize=10)  # Smaller tick size

        # Adding colorbar for edge colors if applicable
        if edge_data is not None:
            edge_cbar_ax = fig.add_axes([0.97, 0.15, 0.015, 0.4])  # Adjust these values as needed
            sm_edge = plt.cm.ScalarMappable(cmap=edge_cmap, norm=edge_norm)
            sm_edge.set_array([])
            cbar_edge = plt.colorbar(sm_edge, cax=edge_cbar_ax, orientation='vertical')
            cbar_edge.set_label(edge_label, fontsize=10)  # Smaller label size
            cbar_edge.ax.tick_params(labelsize=10)  # Smaller tick size
    if is_show:
        plt.show()
    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        
def plot_pos_neg_dots(positive, negative, years, marker_size=100, marker_type='o', alpha=1.0, tick_font_name='Arial', 
                tick_font_size=12, positive_color='blue', negative_color='red', title='Example Dot Plot', 
                xlabel='Year', ylabel='Value', y_limits=None, fig_size=(10, 5), is_show=True, is_legend=False, 
                positive_label='Positive', negative_label='Negative', save_path=None):
    """
    Plots positive and negative values on a dot plot with customizable properties including marker type and legend labels.

    Args:
    positive (array-like): Array of positive values.
    negative (array-like): Array of negative values.
    years (array-like): Array of years corresponding to the values.
    marker_size (int, optional): Size of the markers. Defaults to 100.
    marker_type (str, optional): Type of the marker. Defaults to 'o' (circle).
    alpha (float, optional): Transparency of the markers. Defaults to 1.0.
    tick_font_name (str, optional): Font name for the tick labels. Defaults to 'Arial'.
    tick_font_size (int, optional): Font size for the tick labels. Defaults to 12.
    positive_color (str, optional): Color for positive value markers. Defaults to 'blue'.
    negative_color (str, optional): Color for negative value markers. Defaults to 'red'.
    title (str, optional): Title of the plot. Defaults to 'Example Dot Plot'.
    xlabel (str, optional): Label for the x-axis. Defaults to 'Year'.
    ylabel (str, optional): Label for the y-axis. Defaults to 'Value'.
    y_limits (tuple, optional): Tuple of (min, max) for y-axis limits. If None, defaults to [-25, 25].
    fig_size (tuple, optional): Figure size. Defaults to (10, 5).
    is_show (bool, optional): Whether to show the plot. Defaults to True.
    is_legend (bool, optional): Whether to display a legend. Defaults to False.
    positive_label (str, optional): Legend label for positive values. Defaults to 'Positive'.
    negative_label (str, optional): Legend label for negative values. Defaults to 'Negative'.
    save_path (str, optional): Path to save the plot image file. If None, the plot is not saved. Defaults to None.
    """
    fig, ax = plt.subplots(figsize=fig_size)

    # Plot positive values
    for i, val in enumerate(positive):
        y_positions = np.linspace(0, val - 1, val)
        ax.scatter([years[i]] * val, y_positions, color=positive_color, s=marker_size, marker=marker_type, 
                   alpha=alpha, label=positive_label if i == 0 and is_legend else "")

    # Plot negative values
    for i, val in enumerate(negative):
        y_positions = np.linspace(0, abs(val) - 1, abs(val))
        ax.scatter([years[i]] * abs(val), -y_positions, color=negative_color, s=marker_size, marker=marker_type, 
                   alpha=alpha, label=negative_label if i == 0 and is_legend else "")

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(y_limits if y_limits else [-25, 25])

    # Setting tick font properties
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname(tick_font_name)
        label.set_fontsize(tick_font_size)

    if is_legend:
        ax.legend()

    if save_path:
        plt.savefig(save_path)

    if is_show:
        plt.show()
    else:
        plt.close()
        
def plot_clustered_data(df, columns, x_index, y_index, n_clusters=3, method='KMeans',
                        marker_size=50, marker_colors=None, marker_type='o',
                        fig_size=(10, 5), tick_font_size=12, tick_font_name='Arial',
                        xlabel='X-axis', ylabel='Y-axis', label_font_size=14,
                        is_legend=True, cluster_names=None, is_show=True, save_path=None,
                        is_boundary=False, boundary_color='black', boundary_linewidth=2, boundary_alpha=0.5):
    """
    Plots clustered data from a DataFrame using specified columns and clustering parameters, with optional smooth boundaries.

    Args:
    df (pandas.DataFrame): The DataFrame containing the data.
    columns (list): List of column indices or names to include in the clustering.
    x_index (int or str): Index or name of the column to use for the x-axis.
    y_index (int or str): Index or name of the column to use for the y-axis.
    n_clusters (int, optional): Number of clusters for the clustering algorithm (not applicable to DBSCAN). Default is 3.
    method (str, optional): Clustering method; 'KMeans', 'Agglomerative', 'DBSCAN', 'Spectral', or 'GaussianMixture'. Default is 'KMeans'.
    marker_size (int, optional): Size of the markers in the plot. Default is 50.
    marker_colors (list or dict, optional): Colors for the markers, can be a list or dict assigning colors per cluster. If None, defaults to 'viridis' colormap.
    marker_type (str, optional): Type of marker (e.g., 'o', 'x', '^'). Default is 'o'.
    fig_size (tuple, optional): Size of the figure. Default is (10, 5).
    tick_font_size (int, optional): Font size for tick labels. Default is 12.
    tick_font_name (str, optional): Font name for tick labels. Default is 'Arial'.
    xlabel (str, optional): Label for the x-axis. Default is 'X-axis'.
    ylabel (str, optional): Label for the y-axis. Default is 'Y-axis'.
    label_font_size (int, optional): Font size for axis labels. Default is 14.
    is_legend (bool, optional): Whether to show legend. Default is True.
    cluster_names (list, optional): Custom names for each cluster in the legend. If None, default names are used.
    is_show (bool, optional): Whether to show the plot. Default is True.
    save_path (str, optional): Path to save the plot if specified.
    is_boundary (bool, optional): Whether to draw smooth convex hull boundary around each cluster. Default is False.
    boundary_color (str, optional): Color of the boundary line. Default is 'black'.
    boundary_linewidth (int, optional): Line width of the boundary. Default is 2.
    boundary_alpha (float, optional): Alpha (transparency) of the boundary line. Default is 0.5.

    Returns:
    None
    """
    # Ensure the column indices are correct
    if isinstance(columns[0], str):
        data = df[columns].copy()
    else:
        data = df.iloc[:, columns].copy()
        
    if isinstance(x_index, str):
        x_data = df[x_index]
    else:
        x_data = df.iloc[:, x_index]

    if isinstance(y_index, str):
        y_data = df[y_index]
    else:
        y_data = df.iloc[:, y_index]
    
    # Normalize the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # Clustering
    if method == 'KMeans':
        model = KMeans(n_clusters=n_clusters)
    elif method == 'Agglomerative':
        model = AgglomerativeClustering(n_clusters=n_clusters)
    elif method == 'DBSCAN':
        model = DBSCAN()
    elif method == 'Spectral':
        model = SpectralClustering(n_clusters=n_clusters)
    elif method == 'GaussianMixture':
        model = GaussianMixture(n_components=n_clusters)
    else:
        raise ValueError("Unsupported clustering method. Choose from 'KMeans', 'Agglomerative', 'DBSCAN', 'Spectral', or 'GaussianMixture'.")

    # Fit model
    if method == 'GaussianMixture':
        model.fit(data_scaled)
        labels = model.predict(data_scaled)
    else:
        labels = model.fit_predict(data_scaled)
    
    # Plotting
    fig, ax = plt.subplots(figsize=fig_size)
    unique_labels = np.unique(labels)
    
    if marker_colors is None:
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))
        marker_colors = {label: color for label, color in zip(unique_labels, colors)}
    elif isinstance(marker_colors, list):
        marker_colors = {label: color for label, color in zip(unique_labels, marker_colors)}
    
    for label in unique_labels:
        idx = labels == label
        cluster_label = cluster_names[label] if cluster_names and len(cluster_names) > label else f'Cluster {label}'
        points = ax.scatter(x_data[idx], y_data[idx], c=[marker_colors[label]] * np.sum(idx), s=marker_size, label=cluster_label, marker=marker_type)

        if is_boundary and np.sum(idx) > 2:
            hull = ConvexHull(np.column_stack((x_data[idx], y_data[idx])))
            # Get the convex hull vertices
            hull_points = np.column_stack((x_data[idx], y_data[idx]))[hull.vertices]
            # Close the loop
            hull_points = np.append(hull_points, [hull_points[0]], axis=0)
            # Interpolate to smooth
            t = np.arange(hull_points.shape[0])
            ti = np.linspace(t[0], t[-1], 10 * t.size)  # Interpolation factor of 10
            xi = interp1d(t, hull_points[:, 0], kind='cubic')(ti)
            yi = interp1d(t, hull_points[:, 1], kind='cubic')(ti)
            ax.plot(xi, yi, color=boundary_color, linewidth=boundary_linewidth, alpha=boundary_alpha)

    ax.set_xlabel(xlabel, fontsize=label_font_size, fontname=tick_font_name)
    ax.set_ylabel(ylabel, fontsize=label_font_size, fontname=tick_font_name)
    plt.xticks(fontsize=tick_font_size, fontname=tick_font_name)
    plt.yticks(fontsize=tick_font_size, fontname=tick_font_name)
    
    if is_legend:
        ax.legend(title="Clusters")
    
    if save_path:
        plt.savefig(save_path)
    
    if is_show:
        plt.show()
    else:
        plt.close()
        
def plot_heatmap_on_geomap(data, top_left_lat, top_left_lon, bottom_right_lat, bottom_right_lon, 
                           threshold, cmap='jet', map_choice='base', zoom=10, is_show=True, save_path=None,
                           title = None, colorbar_label = None,
                           fig_size=(8, 6), x_tick_interval=None, y_tick_interval=None, tick_format="{:.2f}"):
    """
    Plots a heatmap on a geographic map with customizable tick labels.
    """
    # Set up figure and axes
    fig, ax = plt.subplots(figsize=fig_size)

    # Generate grid of coordinates
    lons = np.linspace(top_left_lon, bottom_right_lon, data.shape[1])
    lats = np.linspace(top_left_lat, bottom_right_lat, data.shape[0])
    x, y = np.meshgrid(lons, lats)

    # Apply transparency threshold
    alpha = np.ones_like(data)
    alpha[data <= threshold] = 0

    # Map plotting
    if map_choice == 'base':
        m = Basemap(projection='merc', llcrnrlat=bottom_right_lat, urcrnrlat=top_left_lat,
                    llcrnrlon=top_left_lon, urcrnrlon=bottom_right_lon, resolution='i', ax=ax)
        x, y = m(lons, lats)
        heatmap = m.pcolormesh(x, y, data, shading='auto', cmap=cmap, alpha=alpha)
        m.drawcoastlines()
        m.drawcountries()

        # Custom format function
        def custom_fmt(value):
            return tick_format.format(value)

        # Draw meridians and parallels with custom labels
        if x_tick_interval:
            m.drawmeridians(np.arange(round(top_left_lon), round(bottom_right_lon), x_tick_interval), labels=[0,0,0,1], fmt=custom_fmt)
        if y_tick_interval:
            m.drawparallels(np.arange(round(bottom_right_lat), round(top_left_lat), y_tick_interval), labels=[1,0,0,0], fmt=custom_fmt)

    elif map_choice == 'osm':
        heatmap = ax.pcolormesh(x, y, data, cmap=cmap, alpha=alpha, shading='auto')
        ctx.add_basemap(ax, crs="EPSG:4326", source=ctx.providers.OpenStreetMap.Mapnik, zoom=zoom)
        ax.set_xlim([top_left_lon, bottom_right_lon])
        ax.set_ylim([top_left_lat, bottom_right_lat])
    
    colorbar_label = colorbar_label if colorbar_label is not None else 'Data Intensity'
    title = title if title is not None else 'Heatmap Overlay on Geomap'

    # Add color bar and title
    plt.colorbar(heatmap, label=colorbar_label)
    plt.title(title)

    # Display or save
    if is_show:
        plt.show()
    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        
def plot_quadrant_data(data, x_threshold, y_threshold, category_names=None, xlabel='Centroid offset (mm)', ylabel='Edge height difference (mm)',
                       title='Classification of Data Relative to Threshold Lines', xlabel_size=12, ylabel_size=12, title_size=14, fig_size=(10, 8),
                       marker_color='viridis', marker_size=100, x_tick_interval=None, y_tick_interval=None,
                       tick_font='Arial', tick_font_size=10, is_show=True, is_legend=True, save_path=None):
    """
    Plots the classified data with different colors for each category, with customizable visual features.
    
    :param data: DataFrame with 'x', 'y' columns.
    :param x_threshold: Threshold value for the x-axis.
    :param y_threshold: Threshold value for the y-axis.
    :param fig_size: Size of the figure.
    :param category_names: Custom names for the categories.
    :param xlabel: Label for the x-axis.
    :param ylabel: Label for the y-axis.
    :param title: Title of the plot.
    :param xlabel_size, ylabel_size, title_size: Font sizes for the axis labels and title.
    :param marker_color: Base color map or list of colors for the markers.
    :param marker_size: Size of the markers.
    :param x_tick_interval, y_tick_interval: Custom interval for the x and y ticks.
    :param tick_font, tick_font_size: Font and size for the tick labels.
    :param is_show: If True, display the plot.
    :param is_legend: If True, display the legend.
    :param save_path: If provided, save the plot to this path.
    """
    conditions = [
        (data['x'] < x_threshold) & (data['y'] > y_threshold),
        (data['x'] > x_threshold) & (data['y'] > y_threshold),
        (data['x'] < x_threshold) & (data['y'] < y_threshold),
        (data['x'] > x_threshold) & (data['y'] < y_threshold),
        (data['x'] == x_threshold) | (data['y'] == y_threshold)
    ]
    if category_names is None:
        category_names = ['Above Left', 'Above Right', 'Below Left', 'Below Right', 'On Line']
    data['Category'] = np.select(conditions, category_names, default='On Line')
    
    fig, ax = plt.subplots(figsize=fig_size)
    categories = np.unique(data['Category'])
    
    if isinstance(marker_color, str):
        colors = plt.cm.get_cmap(marker_color)(np.linspace(0, 1, len(categories)))
    else:
        colors = marker_color
    
    for category, color in zip(categories, colors):
        subset = data[data['Category'] == category]
        ax.scatter(subset['x'], subset['y'], s=marker_size, c=[color], label=category)
    
    # Draw threshold lines
    x_min, x_max = data['x'].min(), data['x'].max()
    y_min, y_max = data['y'].min(), data['y'].max()
    x_delta, y_delta = x_max - x_min, y_max - y_min
    ax.axhline(y=y_threshold, color='gray', linestyle='--')
    ax.axvline(x=x_threshold, color='gray', linestyle='--')
    ax.set_xlim(x_min - 0.1 * x_delta, x_max + 0.1 * x_delta)
    ax.set_ylim(y_min - 0.1 * y_delta, y_max + 0.1 * y_delta)
    ax.set_xlabel(xlabel, fontsize=xlabel_size)
    ax.set_ylabel(ylabel, fontsize=ylabel_size)
    ax.set_title(title, fontsize=title_size)

    if x_tick_interval is not None:
        ax.xaxis.set_major_locator(plt.MultipleLocator(x_tick_interval))
    if y_tick_interval is not None:
        ax.yaxis.set_major_locator(plt.MultipleLocator(y_tick_interval))

    plt.xticks(fontsize=tick_font_size, fontname=tick_font)
    plt.yticks(fontsize=tick_font_size, fontname=tick_font)

    if is_legend:
        ax.legend(title='Category')
    if save_path:
        plt.savefig(save_path, dpi=600)
    if is_show:
        plt.show()
        
def plot_ridgelines(data, categories, x_label, title, cmap=None, tick_interval=None, tick_size=10, tick_font='Arial',
                          category_size=14, category_font='Arial', title_size=16, save_path=None, is_show=True, is_legend=True, fig_size=(10, 6)):
    """
    Creates a ridgeline plot from the given data using updated bandwidth parameters.

    :param data: Dictionary or DataFrame containing the data for each category.
    :param categories: List of categories (e.g., months).
    :param x_label: Label for the x-axis.
    :param title: Title of the plot.
    :param cmap: Color palette to use. If None, use a black and white scheme.
    :param tick_interval: Interval for x-axis ticks. If None, use default.
    :param tick_size: Font size for ticks.
    :param tick_font: Font family for ticks.
    :param category_size: Font size for category labels.
    :param category_font: Font family for category labels.
    :param title_size: Font size for the plot title.
    :param save_path: Path to save the plot image, if any.
    :param is_show: Whether to display the plot.
    :param is_legend: Whether to display a legend.
    :param fig_size: Size of the figure (width, height).
    """
    # If the data is a dictionary, convert it to a DataFrame
    if isinstance(data, dict):
        all_data = []
        for category, values in data.items():
            for value in values:
                all_data.append({'Category': category, 'Value': value})
        data = pd.DataFrame(all_data)
    else:
        data.columns = ['Category', 'Value']

    # Set the style and palette
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0), "font.size": tick_size, "font.family": tick_font})
    if cmap is None:
        pal = ['black' if i % 2 == 0 else 'gray' for i in range(len(categories))]
    else:
        pal = sns.color_palette(cmap, len(categories))

    # Create the FacetGrid
    g = sns.FacetGrid(data, row='Category', hue='Category', aspect=15, height=0.75, palette=pal)
    
    # Map the densities using the updated bw_adjust parameter
    g.map(sns.kdeplot, 'Value', clip_on=False, shade=True, alpha=1, lw=1.5, bw_adjust=0.5)
    g.map(sns.kdeplot, 'Value', clip_on=False, color="w", lw=2, bw_adjust=0.5)
    g.map(plt.axhline, y=0, lw=2, clip_on=False)

    # Define and use a simple function to label the plot in axes coordinates
    def label(x, color, label):
        ax = plt.gca()
        ax.text(0, 0.2, label, fontweight="bold", color=color, fontsize=category_size, fontname=category_font,
                ha="left", va="center", transform=ax.transAxes)

    g.map(label, "Value")

    # Set the subplots to overlap and adjust the x-label
    g.figure.subplots_adjust(hspace=-0.25)
    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.despine(bottom=True, left=True)
    g.fig.set_size_inches(fig_size)
    g.figure.suptitle(title, fontsize=title_size)
    g.figure.subplots_adjust(top=0.95)
    g.set_xlabels(x_label)

    # Customize x-tick interval
    if tick_interval is not None:
        plt.gca().xaxis.set_major_locator(plt.MultipleLocator(tick_interval))

    # Add legend
    if is_legend:
        g.add_legend()

    # Save the plot
    if save_path:
        plt.savefig(save_path)

    # Show or hide the plot
    if is_show:
        plt.show()
    else:
        plt.close()

def plot_bins_with_cdf(data, cat_key_name='city', val_key_name='profit', cum_key_name=None, flip_axes=False,
                       bar_color='blue', line_color='orange', marker_type='o', marker_face_color='white', marker_size=10,
                       title='Outlets Profit Analysis', title_size=16, title_font='Arial',
                       xlabel='Profit ($)', ylabel='Cumulative Percentage (%)', label_font='Arial', label_size=14,
                       tick_font='Arial', tick_size=12, legend_loc='upper center', fig_size=(10, 8), is_legend=True, is_show=True, save_path=None):
    """
    Creates a bar chart with an overlaid line chart for bins and CDFs, mainly used for profit analysis, with adjusted axes and line extension.

    :param data: DataFrame or list of dictionaries with specified category and value keys.
    :param cat_key_name: Key for categorical data in DataFrame or dict.
    :param val_key_name: Key for value data in DataFrame or dict.
    :param cum_key_name: Key for cumulative data in DataFrame or dict, if pre-calculated.
    :param flip_axes: Boolean, flips the axes for horizontal view.
    :param bar_color, line_color: Colors for the bar and line plots.
    :param marker_type, marker_face_color, marker_size: Customizations for markers in the line plot.
    :param title, title_size, title_font: Customizations for the plot title.
    :param xlabel, ylabel, label_font, label_size: Customizations for the axes labels.
    :param tick_font, tick_size: Font customizations for ticks.
    :param legend_loc: Location of the legend.
    :param fig_size: Figure size.
    :param is_legend: Toggle display of legend.
    :param is_show: Toggle display of the plot.
    :param save_path: Path to save the plot image, if specified.
    """
    if isinstance(data, list):
        data = pd.DataFrame(data)

    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    fig, ax1 = plt.subplots(figsize=fig_size)

    bar_positions = np.arange(len(data[cat_key_name]))  # positions for the bars

    if flip_axes:
        ax1.barh(bar_positions, data[val_key_name], color=bar_color, label='Profit')
        ax1.set_yticks(bar_positions)
        ax1.set_yticklabels(data[cat_key_name], fontsize=tick_size, fontname=tick_font)
        ax1.set_xlabel(xlabel, fontsize=label_size, fontname=label_font)
        ax2 = ax1.twiny()
        ax2.set_xlim(0, 105)  # Align zero positions of two axes
    else:
        ax1.bar(bar_positions, data[val_key_name], color=bar_color, label='Profit')
        ax1.set_xticks(bar_positions)
        ax1.set_xticklabels(data[cat_key_name], rotation=45, ha='right', fontsize=tick_size, fontname=tick_font)
        ax1.set_ylabel(xlabel, fontsize=label_size, fontname=label_font)
        ax2 = ax1.twinx()
        ax2.set_ylim(0, 105)  # Ensure y-axis for cumulative starts at 0

    extended_positions = np.append(bar_positions, bar_positions[-1] + 1)  # Extend line for visual improvement

    if cum_key_name is not None:
        extended_cumulative = np.append(data[cum_key_name], data[cum_key_name].iloc[-1])
    else:
        # Calculate cumulative percentage if not provided
        extended_cumulative = [sum(data[val_key_name][:i+1]) / sum(data[val_key_name]) * 100 for i in range(len(data[val_key_name]))]
        extended_cumulative.append(extended_cumulative[-1])

    # Plot the extended line for cumulative data
    if flip_axes:
        ax2.plot(extended_cumulative[-2:], extended_positions[-2:]- 0.5, '-', color=line_color, markerfacecolor=marker_face_color,
                 markersize=marker_size, label='Cumulative %')
        ax2.plot(extended_cumulative[:-1], extended_positions[:-1] - 0.5, marker_type + '-', color=line_color, markerfacecolor=marker_face_color,
                 markersize=marker_size, label='Cumulative %')
        ax2.fill_betweenx(extended_positions - 0.5, 0, extended_cumulative, color=line_color, alpha=0.1)
    else:
        ax2.plot(extended_positions[-2:]- 0.5, extended_cumulative[-2:], '-', color=line_color, markerfacecolor=marker_face_color,
                 markersize=marker_size, label='Cumulative %')
        ax2.plot(extended_positions[:-1] - 0.5, extended_cumulative[:-1], marker_type + '-', color=line_color, markerfacecolor=marker_face_color,
                 markersize=marker_size, label='Cumulative %')
        ax2.fill_between(extended_positions - 0.5, 0, extended_cumulative, color=line_color, alpha=0.1)
    
    # if flip_Axes, we also need to change the order of y reversely
    if flip_axes:
       plt.gca().invert_yaxis()
    
    ax2.set_ylabel(ylabel, fontsize=label_size, fontname=label_font)

    # Set title and layout
    ax1.set_title(title, fontsize=title_size, fontname=title_font)
    plt.tight_layout()
    # print(ax2.get_legend_handles_labels())
    # Add legend
    if is_legend:
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = [ax2.get_legend_handles_labels()[0][1]], [ax2.get_legend_handles_labels()[1][1]]
        ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc=legend_loc)

    # Optionally save the plot
    if save_path:
        plt.savefig(save_path)

    # Display or hide the plot based on is_show
    if is_show:
        plt.show()
    else:
        plt.close()
        
def plot_surface_with_residuals(x_data, y_data, z_data,
                         xlabel='X', ylabel='Y', zlabel='Z',
                         x_tick_interval=None, y_tick_interval=None, z_tick_interval=None,
                         tick_fontsize=10,
                         line_color='orange', line_thickness=2,
                         dot_color='red', dot_size=5,
                         surface_cmap='viridis', alpha=0.5,
                         is_legend=True, legend_loc='upper right',
                         is_show=True, save_path=None):
    """
    Fits a polynomial surface to the given x_data, y_data, z_data, and plots the surface with residuals.
    
    :param x_data, y_data, z_data: Coordinates of the data points.
    :param xlabel, ylabel, zlabel: Labels for the axes.
    :param x_tick_interval, y_tick_interval, z_tick_interval: Interval for the ticks on the x, y, and z axes.
    :param tick_fontsize: Font size for the tick labels.
    :param line_color, dot_color: Colors for the lines and dots.
    :param line_thickness, dot_size: Thickness of lines and size of dots.
    :param surface_cmap: Color map for the surface plot.
    :param alpha: Transparency of the surface plot.
    :param is_legend: Boolean to toggle legend display.
    :param legend_loc: Location of the legend.
    :param is_show: Boolean to toggle display of the plot.
    :param save_path: Path to save the plot image.
    """
    def __polynomial_surface(xy, a, b, c, d, e, f):
        x, y = xy
        return a + b*x + c*y + d*x**2 + e*y**2 + f*x*y
    
    # Prepare data for fitting
    xy_data = np.vstack((x_data, y_data))
    
    # Fit the model to the data
    popt, pcov = curve_fit(__polynomial_surface, xy_data, z_data, maxfev=10000)
    
    # Generate data for the surface plot
    x_range = np.linspace(min(x_data), max(x_data), 30)
    y_range = np.linspace(min(y_data), max(y_data), 30)
    x_grid, y_grid = np.meshgrid(x_range, y_range)
    z_grid = __polynomial_surface(np.vstack((x_grid.ravel(), y_grid.ravel())), *popt).reshape(x_grid.shape)
    
    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the surface
    ax.plot_surface(x_grid, y_grid, z_grid, cmap=surface_cmap, alpha=alpha)
    
    # Plot the data points
    ax.scatter(x_data, y_data, z_data, color=dot_color, s=dot_size)
    
    # Calculate and plot residuals
    z_pred = __polynomial_surface(xy_data, *popt)
    residuals = z_data - z_pred
    for (xi, yi, zi, ri) in zip(x_data, y_data, z_data, residuals):
        ax.plot([xi, xi], [yi, yi], [zi, zi - ri], color=line_color, linewidth=line_thickness)

    # Set labels
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    
    # Set tick intervals if specified
    if x_tick_interval:
        ax.xaxis.set_major_locator(plt.MultipleLocator(x_tick_interval))
    if y_tick_interval:
        ax.yaxis.set_major_locator(plt.MultipleLocator(y_tick_interval))
    if z_tick_interval:
        ax.zaxis.set_major_locator(plt.MultipleLocator(z_tick_interval))

    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    
    # Create custom legend handles
    custom_lines = [Line2D([0], [0], color=dot_color, marker='o', linestyle='None', markersize=np.sqrt(dot_size)),
                    Line2D([0], [0], color=line_color, lw=line_thickness)]
    
    # Add legend
    if is_legend:
        ax.legend(custom_lines, ['Data Points', 'Residuals'], loc=legend_loc)
    
    # Layout
    plt.tight_layout()

    # Optionally save the plot
    if save_path:
        plt.savefig(save_path)

    # Display or hide the plot
    if is_show:
        plt.show()
    else:
        plt.close()

def plot_ternary(data, labels, scale=100, tick_interval=10, color_map='viridis', 
                 title = "Spruce Composition Analysis", title_font_size = 10,
                 fig_size = (10, 8), label_font_size = 10, label_offset = 0.10, 
                 tick_font_size = 10, tick_offset = 0.01, is_legend = True,
                 is_show = True, save_path = None):
    """
    Plots a ternary diagram with adjusted tick label positions.
    """
    # Setup the plot
    figure, tax = ternary.figure(scale=scale)
    tax.set_title(title, fontsize=title_font_size)
    figure.set_size_inches(fig_size[0], fig_size[1])
    tax.boundary(linewidth=2.0)
    tax.gridlines(multiple=10, color="black")

    # Axis labels and title

    tax.left_axis_label(labels[0], fontsize=label_font_size, offset=label_offset)
    tax.right_axis_label(labels[1], fontsize=label_font_size, offset=label_offset)
    tax.bottom_axis_label(labels[2], fontsize=label_font_size, offset=label_offset)

    # Dynamic colors for points
    points = {}
    for (a, b, c, label) in data:
        if label not in points:
            points[label] = []
        points[label].append((a * scale, b * scale, c * scale))
    
    cmap = plt.get_cmap(color_map)
    unique_labels = set([label for (_, _, _, label) in data])
    color_dict = {label: cmap(i / (len(unique_labels) - 1)) for i, label in enumerate(unique_labels)}

    for label, pts in points.items():
        tax.scatter(pts, label=label, color=color_dict[label], s=80, edgecolor='k')

    # Manually setting ticks on the axes with adjusted offset to keep within bounds
    tax.ticks(axis='lbr', linewidth=1, multiple=tick_interval, offset=tick_offset, fontsize=tick_font_size)
    if is_legend:
        tax.legend(prop={'size': 8}, loc='upper left')
    tax.clear_matplotlib_ticks()  # Clear default ticks
    if is_show:
        plt.show()
    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        
def plot_radar_chart(data, fig_size=(6, 6), tick_font_size=10, tick_font_name='Arial',
                     title="Ice Cream Order Analysis", title_font_size=20, title_font_name='Arial',
                     is_legend=True, is_show=True, save_path=None,
                     color_choice='viridis', line_style='-', line_thickness=2,
                     marker_size=5, marker_type='o', marker_color=None):
    """
    Creates a radar chart with dynamic styling options for lines and markers.

    :param data: DataFrame with index as categories (e.g., months) and columns as data series (e.g., flavors).
    :param fig_size, tick_font_size, tick_font_name, title, etc.: Styling parameters for the plot.
    :param color_choice: Color map name or list of colors.
    :param line_style: Style of lines, single value or list.
    :param line_thickness: Thickness of lines, single value or list.
    :param marker_size: Size of markers, single value or list.
    :param marker_type: Type of markers, single value or list.
    :param marker_color: Color of markers, single value or list.
    """
    labels = data.index
    num_vars = len(labels)

    # Calculate angles for the radar chart
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Close the radar plot by returning to the start

    fig, ax = plt.subplots(figsize=fig_size, subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=tick_font_size, fontname=tick_font_name)

    plt.title(title, size=title_font_size, color='black', y=1.1, fontname=title_font_name)

    # Handle dynamic color settings
    if isinstance(color_choice, str):
        cmap = plt.get_cmap(color_choice)
        colors = [cmap(i / len(data.columns)) for i in range(len(data.columns))]
    else:
        colors = color_choice

    # Determine if properties are lists or single values
    def expand_property(prop, length):
        if isinstance(prop, list):
            if len(prop) != length:
                raise ValueError(f"Length of '{prop}' must be the same as the number of data columns")
            return prop
        else:
            return [prop] * length

    # Expand properties for multiple categories
    line_styles = expand_property(line_style, len(data.columns))
    line_thicknesses = expand_property(line_thickness, len(data.columns))
    marker_sizes = expand_property(marker_size, len(data.columns))
    marker_types = expand_property(marker_type, len(data.columns))
    marker_colors = expand_property(marker_color if marker_color else colors, len(data.columns))

    # Plot each category with its specific style
    for idx, (column, style, thickness, msize, mtype, mcolor) in enumerate(zip(data.columns, line_styles, line_thicknesses, marker_sizes, marker_types, marker_colors)):
        values = data[column].tolist()
        values += values[:1]  # Close the data loop
        ax.plot(angles, values, mtype + style, linewidth=thickness, label=column, color=mcolor, markersize=msize)
        ax.fill(angles, values, color=mcolor, alpha=0.25)

    if is_legend:
        plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    if is_show:
        plt.show()
    else:
        plt.close()
        
def plot_heatmap_with_bound_and_curves(data, boundary_data, condition_lines, condition_labels,
                                       fig_size=(10, 7), cmap='viridis', colorbar_label='Engine Performance (HP)',
                                       boundary_color='red', boundary_linewidth=2, line_styles=None, line_colors=None,
                                       line_widths=None, title="Baseline Torque Speed Map", title_size=16,
                                       xlabel='Engine Speed (rpm)', ylabel='Torque (lb-ft)', label_font_size=12,
                                       tick_font_size=10, is_legend=True, is_show=True, save_path=None):
    """
    Plots a contour plot for performance data limited by a boundary curve and overlays condition lines with annotations.
    
    :param data: DataFrame with columns 'x', 'y', 'z' for plotting.
    :param boundary_data: DataFrame with 'x', 'y' defining the boundary.
    :param condition_lines: List of DataFrames each containing 'x', 'y' for condition lines.
    :param condition_labels: List of strings for labeling each condition line.
    :param fig_size: Tuple of figure size.
    :param cmap: Colormap for the heatmap.
    :param colorbar_label: Label for the color bar.
    :param boundary_color, boundary_linewidth: Style for the boundary line.
    :param line_styles, line_colors, line_widths: Styles for each condition line.
    :param title, title_size: Plot title and font size.
    :param xlabel, ylabel, label_font_size: Axis labels and font size.
    :param tick_font_size: Font size for the tick labels.
    :param is_legend: Whether to display the legend.
    :param is_show: If True, show the plot, otherwise save it.
    :param save_path: If provided, save the plot to this path.
    """
    fig, ax = plt.subplots(figsize=fig_size)

    # Define grid and interpolate z values
    x = np.linspace(min(data['x']), max(data['x']), 100)
    y = np.linspace(min(data['y']), max(data['y']), 100)
    X, Y = np.meshgrid(x, y)
    Z = griddata((data['x'], data['y']), data['z'], (X, Y), method='linear')

    # Generate contour plot
    contour = ax.contourf(X, Y, Z, levels=15, cmap=cmap, alpha=0.6, vmin=min(data['z']), vmax=max(data['z']))
    cbar = fig.colorbar(contour)
    cbar.set_label(colorbar_label, fontsize = label_font_size)

    # Plot boundary
    ax.plot(boundary_data['x'], boundary_data['y'], color=boundary_color, linewidth=boundary_linewidth, label='Boundary')
    boundary_data['x'] = boundary_data['x'].tolist() + [max(data['x']), min(data['x'])]
    boundary_data['y'] = boundary_data['y'].tolist() + [min(data['y']), min(data['y'])]
    # Clip the contour inside the boundary
    path = mpath.Path(list(zip(boundary_data['x'], boundary_data['y'])))
    patch = patches.PathPatch(path, visible=False)
    ax.add_patch(patch)
    for collection in contour.collections:
        collection.set_clip_path(patch)

    # Ensure that lists for line properties are set up correctly
    if line_styles is None:
        line_styles = ['-'] * len(condition_lines)
    if line_colors is None:
        line_colors = ['black'] * len(condition_lines)
    if line_widths is None:
        line_widths = [2] * len(condition_lines)
        
    for line, label, style, color, width in zip(condition_lines, condition_labels, line_styles, line_colors, line_widths):
        ax.plot(line['x'], line['y'], style, color=color, linewidth=width)
        point_index = round(len(line['x']) * 0.95)  # Position for annotation
        ax.annotate(label, xy=(line['x'][point_index], line['y'][point_index]), xytext=(10, 5),
                    textcoords='offset points', ha='center', va='bottom', fontsize=9, color=color, weight='bold')

    # Set labels and title
    ax.set_xlabel(xlabel, fontsize=label_font_size)
    ax.set_ylabel(ylabel, fontsize=label_font_size)
    ax.set_title(title, size=title_size)
    ax.set_xticklabels(ax.get_xticks(), fontsize=tick_font_size)
    ax.set_yticklabels(ax.get_yticks(), fontsize=tick_font_size)
    cbar.ax.tick_params(labelsize=tick_font_size)
    # Display the legend if enabled
    if is_legend:
        ax.legend()

    # Show or save the plot based on user preference
    if is_show:
        plt.show()
    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches='tight')

def plot_timeline(data, title="Historical Timeline", time_resolution='y', label_format=None,
                  time_interval=2, is_grid=True, is_show=True, save_path=None, fig_size=(15, 8),
                  color_palette='tab20', label_color='black', label_font_size=10, label_font_name='Arial'):
    """
    Plots a timeline chart with customizable labeling, time resolution, and other display options.
    """
    fig, ax = plt.subplots(figsize=fig_size)
    
    # Fetch the colormap without specifying the length
    cmap = colormaps[color_palette]

    # Parse only if necessary and sort by start time
    data = sorted([(parse(start) if isinstance(start, str) else start,
                    parse(end) if isinstance(end, str) else end) for start, end in data], key=lambda x: x[0], reverse=False)

    # Generate and set ticks
    ticks = __generate_tick_times(data, time_resolution, time_interval)
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(DateFormatter('%Y-%m' if time_resolution == 'm' else '%Y'))

    # Plotting data
    for i, (start, end) in enumerate(data):
        width = end - start
        color = cmap(i / len(data))  # Normalize color index over the range of data
        ax.barh(y=i, width=width, left=start, height=0.4, color=color, align='center')
        # Center text on each bar
        ax.text(start + width / 2, i, label_format + f" {i+1}" if isinstance(label_format, str) else '', ha='center', va='center',
                fontsize=label_font_size, color=label_color, fontname=label_font_name)

    ax.set_xlabel("Time")
    plt.title(title)
    
    if is_grid:
        ax.grid(True, axis='x', linestyle='--', linewidth=0.5)
    
    ax.set_yticks([])  # Hide y-axis labels
    ax.invert_yaxis()  # Reverse the order of the y-axis
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if is_show:
        plt.show()

def plot_density_contours(data, title='Density Contour Plot', xlabel='Idle (min.)', ylabel='Erupting (min.)',
                          x_tick_interval=None, y_tick_interval=None, font_name='Arial', font_size=12, 
                          point_color = 'black', point_size = 10,
                          color_map='viridis', levels=15, convert_to_percent=True, fig_size=(10, 8),
                          is_colorbar=True, save_path=None, is_show=True):
    """
    Plots a density contour map for given data with extensive customization.

    Parameters:
    - data (list of tuples or 2D array): Input data where each tuple is a point (x, y).
    - title (str): Title of the plot.
    - xlabel, ylabel (str): Labels for the x-axis and y-axis.
    - point_color (str): Color of the data points.
    - point_size (int): Size of the data points.
    - x_tick_interval, y_tick_interval (float): Spacing between ticks on x-axis and y-axis.
    - font_name (str): Font name for text elements.
    - font_size (int): Font size for text elements.
    - color_map (str): Matplotlib colormap name.
    - levels (int): Number of contour levels.
    - convert_to_percent (bool): If True, convert density values to percentages.
    - fig_size (tuple): Dimensions of the figure (width, height).
    - is_colorbar (bool): If True, display a colorbar.
    - save_path (str): If provided, the plot is saved to this path.
    - is_show (bool): If True, display the plot window.
    """
    fig, ax = plt.subplots(figsize=fig_size)
    # colors = colormaps.get_cmap(color_map, len(data))

    # Extract x and y coordinates from the data
    x = np.array([point[0] for point in data])
    y = np.array([point[1] for point in data])
    
    # Calculate the point density
    xy = np.vstack([x, y])
    density = gaussian_kde(xy)(xy)
    if convert_to_percent:
        density = density / density.sum() * 100  # Convert density to percentages
    
    # Sort the points by density
    idx = density.argsort()
    x, y, density = x[idx], y[idx], density[idx]
    
    

    # Set up the grid for contours
    xi, yi = np.linspace(x.min(), x.max(), 100), np.linspace(y.min(), y.max(), 100)
    xi, yi = np.meshgrid(xi, yi)
    zi = gaussian_kde(xy)(np.vstack([xi.flatten(), yi.flatten()]))

    # Plot contour lines
    contour = ax.contourf(xi, yi, zi.reshape(xi.shape), levels=levels, cmap=color_map, alpha=0.6)
    if is_colorbar:
        cbar = plt.colorbar(contour, ax=ax, label='Density (%)' if convert_to_percent else 'Density')
        cbar.ax.tick_params(labelsize=font_size - 1)  # set font size for colorbar
        # set font size for colorbar label
        cbar.ax.yaxis.label.set_fontsize(font_size)
    # Scatter plot for the data points
    scatter = ax.scatter(x, y, c=point_color, s=point_size, edgecolor='none', cmap=color_map)
    # Customize the plot
    ax.set_xlabel(xlabel, fontsize=font_size, fontname=font_name)
    ax.set_ylabel(ylabel, fontsize=font_size, fontname=font_name)
    plt.title(title, fontsize=font_size + 2, fontname=font_name)

    if x_tick_interval:
        ax.xaxis.set_major_locator(plt.MultipleLocator(x_tick_interval))
    if y_tick_interval:
        ax.yaxis.set_major_locator(plt.MultipleLocator(y_tick_interval))

    plt.tight_layout()

    # Save the plot if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches='tight')

    # Display the plot if is_show is True
    if is_show:
        plt.show()
    else:
        plt.close()
        

def plot_two_colored_time_series(data, threshold, title='Time Series Plot', xlabel='Time', ylabel='Value',
                                 color_above='red', color_below='black', line_style='-', line_width=2,
                                 threshold_color='gray', threshold_style='--', threshold_width=1,
                                 fig_size=(15, 6), x_tick_interval=None, y_tick_interval=None,
                                 tick_font_name='sans-serif', tick_font_size=12,
                                 is_legend=True, legend_text='Threshold',
                                 save_path=None, is_show=True):
    """
    Plots a time series where the line color changes above and below a certain threshold,
    and segments that cross the threshold are divided into two parts with appropriate colors.
    Line properties and threshold line properties can be customized.

    Parameters:
    - data (pandas Series): Series with a DateTime index and one column of values.
    - threshold (float): The threshold value at which the line color changes.
    - title (str): Title of the plot.
    - xlabel (str), ylabel (str): Labels for the x-axis and y-axis.
    - color_above (str), color_below (str): Colors for the segments above and below the threshold.
    - line_style (str): Style of the line (e.g., '-', '--', '-.', ':').
    - line_width (float): Width of the line.
    - threshold_color (str): Color of the threshold line.
    - threshold_style (str): Style of the threshold line (e.g., '-', '--', '-.', ':').
    - threshold_width (float): Width of the threshold line.
    - fig_size (tuple): Figure size.
    - x_tick_interval (float): Interval between ticks on the x-axis.
    - y_tick_interval (float): Interval between ticks on the y-axis.
    - tick_font_name (str): Font name for the tick labels.
    - tick_font_size (int): Font size for the tick labels.
    - is_legend (bool): If True, displays a legend.
    - legend_text (str): Text for the legend corresponding to the threshold line.
    - save_path (str): If provided, saves the plot to this path.
    - is_show (bool): If True, displays the plot.
    """
    fig, ax = plt.subplots(figsize=fig_size)
    font = FontProperties(family=tick_font_name, size=tick_font_size)

    last_index = data.index[0]
    last_value = data.iloc[0]

    for current_index, current_value in data.iloc[1:].items():
        if (last_value < threshold and current_value >= threshold) or (last_value >= threshold and current_value < threshold):
            ratio = (threshold - last_value) / (current_value - last_value)
            threshold_time = last_index + (current_index - last_index) * ratio
            ax.plot([last_index, threshold_time], [last_value, threshold], color=color_below if last_value < threshold else color_above, linestyle=line_style, linewidth=line_width)
            ax.plot([threshold_time, current_index], [threshold, current_value], color=color_above if current_value >= threshold else color_below, linestyle=line_style, linewidth=line_width)
        else:
            ax.plot([last_index, current_index], [last_value, current_value], color=color_above if last_value >= threshold else color_below, linestyle=line_style, linewidth=line_width)
        last_index = current_index
        last_value = current_value

    # Plotting the threshold line
    threshold_line = ax.axhline(y=threshold, color=threshold_color, linestyle=threshold_style, linewidth=threshold_width)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Set tick intervals and font properties if specified
    if x_tick_interval:
        ax.xaxis.set_major_locator(MultipleLocator(x_tick_interval))
    if y_tick_interval:
        ax.yaxis.set_major_locator(MultipleLocator(y_tick_interval))

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(font)

    # Handling the legend
    if is_legend:
        ax.legend([threshold_line], [legend_text], loc='best')

    if save_path:
        plt.savefig(save_path)

    if is_show:
        plt.show()

def plot_scatter_with_threshold(data, threshold, title='Scatter Plot', xlabel='X', ylabel='Y', edge_color=None,
                                base_color_above='Blues', base_color_below='Reds', fig_size=(15, 6), is_legend=False,
                                legend_font_size=10, legend_font_name='sans-serif',
                                label_font_size=12, label_font_name='sans-serif',
                                line_color='gray', line_style='--', line_thickness=1,
                                save_path=None, is_show=True):
    """
    Plots a scatter plot where the color saturation varies based on the distance from a threshold value and
    uses different color maps for values above and below the threshold. The further from the threshold,
    the more saturated the color.

    Parameters:
    - data (pandas DataFrame): DataFrame with 'x' and 'y' columns.
    - threshold (float): Value at which the threshold line is drawn and colors are based on.
    - title (str): Title of the plot.
    - xlabel (str), ylabel (str): Labels for the x-axis and y-axis.
    - base_color_above, base_color_below (str): Base color maps for points above and below the threshold.
    - fig_size (tuple): Figure size.
    - save_path (str): If provided, saves the plot to this path.
    - is_show (bool): If True, displays the plot.
    - label_font_size (int): Font size for the axis labels.
    - label_font_name (str): Font name for the axis labels.
    - legend_font_size (int): Font size for the legend.
    - legend_font_name (str): Font name for the legend.
    - line_color (str): Color of the threshold line.
    - line_style (str): Style of the threshold line.
    - line_thickness (float): Thickness of the threshold line.
    """
    fig, ax = plt.subplots(figsize=fig_size)
    label_font = FontProperties(family=label_font_name, size=label_font_size)
    legend_font = FontProperties(family=legend_font_name, size=legend_font_size)
    
    # Calculate distances from the threshold and normalize
    distances = np.abs(data['y'] - threshold)
    norm = Normalize(vmin=0, vmax=distances.max())
    
    # Prepare color maps
    cmap_above = plt.get_cmap(base_color_above)
    cmap_below = plt.get_cmap(base_color_below)

    # Split data above and below the threshold
    above_threshold = data['y'] >= threshold
    below_threshold = data['y'] < threshold

    # Plot scatter points with different color maps
    scatter1 = ax.scatter(data['x'][above_threshold], data['y'][above_threshold], c=distances[above_threshold],
                         cmap=cmap_above, norm=norm, edgecolor=edge_color, label='Above Threshold')
    scatter2 = ax.scatter(data['x'][below_threshold], data['y'][below_threshold], c=distances[below_threshold],
               cmap=cmap_below, norm=norm, edgecolor=edge_color, label='Below Threshold')
    
    # Draw the threshold line
    ax.axhline(y=threshold, color=line_color, linestyle=line_style, linewidth=line_thickness)
    # Custom color map for the colorbar
    colors_above = cmap_above(np.linspace(0, 1, 256))
    colors_below = cmap_below(np.linspace(0, 1, 256))
    all_colors = np.vstack((colors_below[::-1], colors_above))
    custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', all_colors)
    sm = ScalarMappable(cmap=custom_cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Distance from Threshold', fontproperties=label_font)

    # Set labels, title, and fonts
    ax.set_title(title, fontproperties=label_font)
    ax.set_xlabel(xlabel, fontproperties=label_font)
    ax.set_ylabel(ylabel, fontproperties=label_font)

    # Set tick labels
    ax.tick_params(axis='both', labelsize=label_font_size)

    # Handling the legend
    if is_legend:
        ax.legend(prop=legend_font)

    if save_path:
        plt.savefig(save_path)

    if is_show:
        plt.show()

def plot_cyclic_bar(data, r0, delta, fig_size=(5, 5), color_list=None, start_angle=90, edge_color='white',
               text_offset=0.02, font_size=10, font_name='sans-serif', label_color='black',
               non_filled_color='lightgrey', is_show=True, save_path=None):
    """
    Draws a circular bar chart with extensive customization options.

    Parameters:
    - data (pd.DataFrame): Data containing values for the chart.
    - r0 (float): Radius of the innermost circle.
    - delta (float): Thickness of each ring.
    - fig_size (tuple): Size of the figure.
    - color_list (list): List of colors for each ring.
    - startangle (int): Starting angle for the first segment.
    - edgecolor (str): Color of the edge between segments.
    - text_offset (float): Vertical offset for the text inside the rings.
    - font_size (int): Font size for labels.
    - font_name (str): Font family for labels.
    - label_color (str): Color of the text labels.
    - non_filled_color (str): Color for the non-filled area of each ring.
    - is_show (bool): If True, displays the plot.
    - save_path (str): If provided, saves the plot to the given path.
    """
    if color_list is None:
        color_list = ['darkcyan', 'red', 'blue', 'green', 'peru', 'yellow', 'magenta', 'tomato', 'maroon', 'purple', 'lime', 'chocolate', 'olive', 'rosybrown', 'gold']

    ind = list(data.index)[::-1]
    num = list(data.iloc[:, 0])[::-1]
    row = data.shape[0]
    labels = data.columns[0]
    col0 = data.iloc[:, 0]
    w = 1.25 * col0.max()  # Maximum value for normalization adjusted to fit the chart within bounds
    
    fig, ax = plt.subplots(figsize=fig_size)
    for i in range(row):
        r1 = delta * row + r0 - delta * i
        colors = [color_list[i % len(color_list)], non_filled_color]
        values = [num[i] / w * 0.7, 0.75 - num[i] / w * 0.7]
        ax.pie(values, shadow=False, colors=colors, startangle=start_angle, radius=r1,
               wedgeprops={'linewidth': 2, 'edgecolor': edge_color})
        plt.text(0, text_offset + r0 + delta * i, ind[i], fontsize=font_size, fontname=font_name, color=label_color, ha='left')
        plt.text(0.8, text_offset + r0 + delta * i, f"{num[i]:.2f}", fontsize=font_size, fontname=font_name, color=label_color, ha='right')

    ax.pie([1], radius=r0, colors='w')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=600)

    if is_show:
        plt.show()
    else:
        plt.close()

def plot_movement_arrows(pre_data, post_data, labels, title='Movement Over Time', xlabel='Population', ylabel='Inequality',
                         x_delta=0.1, y_delta=0.1, arrow_color='red', scale=1, fig_size=(15, 10),
                         use_log_x=False, use_log_y=False, is_show=True, is_grid=False, save_path=None,
                         font_name='Arial', font_size=12, annotation_text_size=10, annotation_text_color='blue'):
    """
    Plots initial points for one dataset and arrows showing movement to another dataset on a potentially log-scaled axes.
    Uses 'quiver' to plot arrows, which is suitable for vector fields.

    Parameters:
    - pre_data (dict): Data points from the initial dataset.
    - post_data (dict): Data points from the subsequent dataset.
    - labels (list): List of labels to annotate.
    - title (str): Title of the plot.
    - xlabel (str): Label for the x-axis.
    - ylabel (str): Label for the y-axis.
    - x_delta, y_delta (float): Margins to adjust the axes limits.
    - arrow_color (str): Color of the arrows.
    - scale (float): Scale factor for the arrows to control their size.
    - fig_size (tuple): Size of the figure.
    - use_log_x, use_log_y (bool): If True, use log scale for the respective axis.
    - is_show (bool): If True, display the plot; otherwise, do not display.
    - is_grid (bool): If True, display the grid.
    - save_path (str): Path to save the figure to file.
    - font_name (str), font_size (int), annotation_text_size (int): Font settings.
    - annotation_text_color (str): Color for the text annotations.
    """
    fig, ax = plt.subplots(figsize=fig_size)
    if use_log_x:
        ax.set_xscale('log')
    if use_log_y:
        ax.set_yscale('log')

    X, Y, U, V, X1, Y1 = [], [], [], [], [], []
    for label in labels:
        x0, y0 = pre_data[label]
        x1, y1 = post_data.get(label, (x0, y0))  # Default to start point if not available
        X.append(x0)
        Y.append(y0)
        X1.append(x1)
        Y1.append(y1)
        U.append(x1 - x0)
        V.append(y1 - y0)
        
        # Plotting initial points
        ax.scatter(x0, y0, color='black')
        ax.text(x1, y1, f'{label}', fontsize=annotation_text_size, color=annotation_text_color,
                verticalalignment='bottom', horizontalalignment='right')

    # Using quiver to draw arrows
    ax.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=scale, color=arrow_color)

    ax.set_title(title, fontsize=font_size, fontname=font_name)
    ax.set_xlabel(xlabel, fontsize=font_size, fontname=font_name)
    ax.set_ylabel(ylabel, fontsize=font_size, fontname=font_name)

    if is_grid:
        ax.grid(True)

    ax.set_xlim(min(min(X), min(X1)) - x_delta, max(max(X), max(X1)) + x_delta)
    ax.set_ylim(min(min(Y), min(Y1)) - y_delta, max(max(Y), max(Y1)) + y_delta)

    if save_path:
        plt.savefig(save_path, dpi=600)

    if is_show:
        plt.show()
        
def plot_rolling_series(data, rolling_period=30, title='Rolling Statistic Time Series', xlabel='Time', ylabel='Value',
                        series_color='gray', max_color='red', min_color='blue', avg_color='black', alpha=0.3,
                        fig_size=(15, 6), line_width=2, line_style='-', legend_labels=None,
                        font_name='Arial', font_size=12, y_lim=None, y_tick_interval=None, x_tick_interval=None,
                        is_grid=True, is_legend=True, is_label=False, save_path=None, is_show=True):
    """
    Plots a time series graph with rolling maximum, minimum, and average values over a specified period.

    Parameters:
    - data (pandas.Series): Time-series data.
    - rolling_period (int): Window size for the rolling calculations.
    - title (str): Title of the plot.
    - xlabel (str), ylabel (str): Labels for the x-axis and y-axis.
    - series_color, max_color, min_color, avg_color (str): Colors for the series and rolling stats.
    - alpha (float): Transparency for the time series data plot.
    - fig_size (tuple): Size of the figure.
    - line_width (int): Width of the lines.
    - line_style (str): Style of the lines (e.g., '-', '--', '-.', ':').
    - legend_labels (dict): Custom labels for the legend.
    - font_name (str): Font name for labels.
    - font_size (int): Font size for labels and ticks.
    - y_lim (tuple): Tuple of (min, max) for the y-axis.
    - y_tick_interval (float): Interval between ticks on the y-axis.
    - x_tick_interval (int): Interval in days between ticks on the x-axis.
    - is_grid (bool): Whether to display a grid.
    - is_legend (bool): Whether to display a legend.
    - is_label (bool): Whether to label the lines.
    - save_path (str): Path to save the plot image if specified.
    - is_show (bool): Whether to display the plot.
    """
    plt.figure(figsize=fig_size)
    ax = plt.gca()

    # Plotting the original time series data
    ax.plot(data.index, data, color=series_color, alpha=alpha, label=legend_labels.get('original', 'Original Data') if is_label else '',
            linestyle=line_style, linewidth=line_width)

    # Rolling calculations
    rolling_max = data.rolling(window=rolling_period, min_periods=1).max()
    rolling_min = data.rolling(window=rolling_period, min_periods=1).min()
    rolling_avg = data.rolling(window=rolling_period, min_periods=1).mean()

    # Plotting rolling statistics
    ax.plot(data.index, rolling_max, color=max_color, label=legend_labels.get('max', 'Rolling Max') if is_label else '',
            linestyle=line_style, linewidth=line_width)
    ax.plot(data.index, rolling_min, color=min_color, label=legend_labels.get('min', 'Rolling Min') if is_label else '',
            linestyle=line_style, linewidth=line_width)
    ax.plot(data.index, rolling_avg, color=avg_color, label=legend_labels.get('avg', 'Rolling Average') if is_label else '',
            linestyle=line_style, linewidth=line_width)

    # Axes and grid
    ax.set_title(title, fontsize=font_size, fontname=font_name)
    ax.set_xlabel(xlabel, fontsize=font_size, fontname=font_name)
    ax.set_ylabel(ylabel, fontsize=font_size, fontname=font_name)
    if y_lim:
        ax.set_ylim(y_lim)
    if y_tick_interval:
        ax.yaxis.set_major_locator(MultipleLocator(y_tick_interval))
    if x_tick_interval:
        ax.xaxis.set_major_locator(DayLocator(interval=x_tick_interval))  # Set day locator to x-axis
        # ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))  # Set the display format for dates

    # Grid and legend
    ax.grid(is_grid)
    if is_legend:
        ax.legend()

    # Setting axis tick properties
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(font_size)
        label.set_fontname(font_name)

    if save_path:
        plt.savefig(save_path)

    if is_show:
        plt.show()

def draw_grouped_polar_bars(groups, group_names, group_colors, label_names = None, fig_size = (8, 8), scale_lim = 100, scale_major = 20, \
                           tick_font_size = 5, label_font_size = 5, save_path = None, is_show = True):
    
    fig = plt.figure(figsize=fig_size, dpi=300, facecolor='white')
    ax = fig.add_subplot(projection='polar')
    
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    
 
    radii = [0]
    colors = ['blue']
    for g, c in zip(groups, group_colors):
        radii.extend(g)
        colors.extend([c]*len(g))
        radii.append(0)
        colors.append('blue')
    radii.pop()
    colors.pop()
    
    N = len(radii)
    
    bottom = 40
    theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
    width = 2 * np.pi / (N + 9)
    
    ax.bar(theta, radii, width=width, bottom=bottom, color=colors)
    s_list, g_no = [], 0

    for t, r in zip(theta, radii):
        if r == 0:
            s_list.append(t)
            if t == 0:
                __scale_value(ax, bottom, t, scale_lim, scale_major, tick_font_size=tick_font_size)
            else:
                __scale(ax, bottom, scale_lim, t, width, scale_major)
        else:
            t2 = np.rad2deg(t)
            if label_names is not None:
                ax.text(t, r + bottom + scale_major*0.5, label_names[g_no],
                        fontsize=label_font_size, rotation=90-t2 if t < np.pi else 270-t2,
                        rotation_mode='anchor', va='center', ha='left' if t < np.pi else 'right',
                        color='black', clip_on=False)
            if g_no == (len(label_names)-1):
                g_no = 0
            else:
                g_no += 1
    
    
    s_list.append(2 * np.pi)
    
    for i in range(len(s_list)-1):
        t = np.linspace(s_list[i]+width, s_list[i+1]-width, 50)
        ax.plot(t, [bottom-scale_major*0.4]*50, linewidth=0.5, color='black')
        ax.text(s_list[i]+(s_list[i+1]-s_list[i])/2,
                bottom-scale_major*1.05, group_names[i], va='center',
                ha='center',fontsize=label_font_size)
    
    ax.set_rlim(0, bottom+scale_lim+scale_major)
    ax.axis('off')
    
    if is_show:
        plt.show()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=600)

def plot_grouped_dot_chart(data, category_col='State', value_cols=None, label_size=12, tick_label_size=10, cmap = 'tab20',
                           font_name='Arial', is_legend=True, is_show=True, fig_size=(10, 8), save_path=None):
    # Dynamically determine value columns if not provided
    if value_cols is None:
        value_cols = [col for col in data.columns if col != category_col]

    # Generate a color palette
    colors = plt.get_cmap(cmap).colors  # This colormap has 20 distinct colors
    color_dict = {category: colors[i % 20] for i, category in enumerate(value_cols)}

    # Create the plot with custom figure size
    fig, ax = plt.subplots(figsize=fig_size)

    # Set font properties
    plt.rcParams.update({'font.size': tick_label_size, 'font.family': font_name})

    # Unique categories for the x-axis
    categories = data[category_col].unique()
    category_positions = np.arange(len(categories))

    # Offset to differentiate the dots within the same category group
    offset = np.linspace(-0.3, 0.3, len(value_cols))

    for idx, value_col in enumerate(value_cols):
        # Positions for each dot
        positions = category_positions + offset[idx]
        # Plot each category
        ax.scatter(positions, data[value_col], color=color_dict[value_col], label=value_col, s=100)
        # Draw vertical lines
        for pos, y in zip(positions, data[value_col]):
            ax.vlines(pos, ymin=0, ymax=y, color=color_dict[value_col], linewidth=1)

    # Set x-ticks and labels with custom sizes
    ax.set_xticks(category_positions)
    ax.set_xticklabels(categories, fontsize=tick_label_size)
    ax.set_xlabel(category_col, fontsize=label_size)
    ax.set_ylabel('Values', fontsize=label_size)
    ax.tick_params(axis='both', which='major', labelsize=tick_label_size)
    ax.set_title(f'Values by {category_col} and Categories', fontsize=label_size + 2)

    # Optionally add legend
    if is_legend:
        ax.legend(title='Categories', title_fontsize=label_size, fontsize=tick_label_size)

    # Optionally save the plot to a file
    if save_path:
        plt.savefig(save_path, format='png', dpi=600)

    # Optionally display the plot
    if is_show:
        plt.show()
    plt.close()

def plot_3d_density_surface(
    data, key_x, key_y,
    grid_size=100, cmap='viridis',
    flip_x=False, flip_y=False,
    xlabel=None, ylabel=None, zlabel='Density',
    title=None, fontsize=12, tick_fontsize=10, tick_pad=2,
    xlim=None, ylim=None, zlim=None,
    x_interval=None, y_interval=None, z_interval=None,
    x_scale_factor=1, y_scale_factor=1,
    figsize=(10, 7), dpi=300,
    save_path=None,
    is_gridline=True, background_color='white',
    legend_bar_size=0.5, is_legend=True
):
    """
    Plots a 3D density surface based on two keys from a list of dictionaries.

    Parameters:
    - data (list of dict): Input data.
    - key_x (str): Key for the x-axis values.
    - key_y (str): Key for the y-axis values.
    - grid_size (int): Resolution of the density surface (default: 100).
    - cmap (str): Colormap for the surface plot.
    - flip_x (bool): If True, reverses the x-axis.
    - flip_y (bool): If True, reverses the y-axis.
    - xlabel (str): Label for x-axis (default: key_x).
    - ylabel (str): Label for y-axis (default: key_y).
    - zlabel (str): Label for z-axis.
    - title (str): Plot title.
    - fontsize (int): Font size for labels.
    - tick_fontsize (int): Font size for axis ticks.
    - tick_pad (int): Padding between tick marks and tick labels.
    - xlim (tuple): Custom x-axis limits (min, max).
    - ylim (tuple): Custom y-axis limits (min, max).
    - zlim (tuple): Custom z-axis limits (min, max).
    - x_interval (float): Interval for x-axis ticks.
    - y_interval (float): Interval for y-axis ticks.
    - z_interval (float): Interval for z-axis and color bar ticks.
    - x_scale_factor (float): Scale factor for x-axis values.
    - y_scale_factor (float): Scale factor for y-axis values.
    - figsize (tuple): Figure size (width, height).
    - dpi (int): Resolution of the figure.
    - save_path (str): If provided, saves the figure to this path.
    - is_gridline (bool): Show or hide dashed, transparent, thin gridlines.
    - background_color (str): Background color of the plot.
    - legend_bar_size (float): Size of the color legend bar.
    - is_legend (bool): Show or hide the color legend.

    Returns:
    - None (Displays the plot or saves if a path is given).
    """

    # Extract x and y values from the list of dictionaries and apply scaling
    x = np.array([d[key_x] for d in data if key_x in d and key_y in d]) / x_scale_factor
    y = np.array([d[key_y] for d in data if key_x in d and key_y in d]) / y_scale_factor

    # Perform Kernel Density Estimation (KDE)
    xy = np.vstack([x, y])
    kde = gaussian_kde(xy)

    # Create a mesh grid for the density estimation
    x_min, x_max = (x.min(), x.max()) if xlim is None else (xlim[0] / x_scale_factor, xlim[1] / x_scale_factor)
    y_min, y_max = (y.min(), y.max()) if ylim is None else (ylim[0] / y_scale_factor, ylim[1] / y_scale_factor)
    x_grid, y_grid = np.linspace(x_min, x_max, grid_size), np.linspace(y_min, y_max, grid_size)
    X, Y = np.meshgrid(x_grid, y_grid)

    # Compute density values
    positions = np.vstack([X.ravel(), Y.ravel()])
    Z = kde(positions).reshape(grid_size, grid_size)

    # If zlim is specified, ensure the color mapping matches
    norm = None
    if zlim:
        Z = np.clip(Z, zlim[0], zlim[1])  # Clip values to zlim range
        norm = mcolors.Normalize(vmin=zlim[0], vmax=zlim[1])

    # Plot the 3D surface
    fig = plt.figure(figsize=figsize, dpi=dpi, facecolor=background_color)
    ax = fig.add_subplot(111, projection='3d')

    # Set background color
    ax.set_facecolor(background_color)

    # Plot surface with proper color normalization
    surf = ax.plot_surface(X, Y, Z, cmap=cmap, edgecolor='none', norm=norm)

    # Customizations
    ax.set_xlabel(xlabel if xlabel else '', fontsize=fontsize, labelpad=tick_pad)
    ax.set_ylabel(ylabel if ylabel else '', fontsize=fontsize, labelpad=tick_pad)
    ax.set_zlabel(zlabel, fontsize=fontsize, labelpad=tick_pad)
    if title:
        ax.set_title(title, fontsize=fontsize + 2)

    # Flip axes if needed
    if flip_x:
        ax.invert_xaxis()
    if flip_y:
        ax.invert_yaxis()

    # Set axis limits if provided
    if xlim:
        ax.set_xlim(xlim[0] / x_scale_factor, xlim[1] / x_scale_factor)
    if ylim:
        ax.set_ylim(ylim[0] / y_scale_factor, ylim[1] / y_scale_factor)
    if zlim:
        ax.set_zlim(zlim)

    # Set tick font sizes and padding
    ax.tick_params(axis='x', labelsize=tick_fontsize, pad=tick_pad)
    ax.tick_params(axis='y', labelsize=tick_fontsize, pad=tick_pad)
    ax.tick_params(axis='z', labelsize=tick_fontsize, pad=tick_pad)

    # Set custom tick intervals for X, Y, and Z axes
    if xlim and x_interval:
        x_ticks = np.arange(xlim[0], xlim[1] + x_interval, x_interval) / x_scale_factor
        ax.set_xticks(x_ticks)
    
    if ylim and y_interval:
        y_ticks = np.arange(ylim[0], ylim[1] + y_interval, y_interval) / y_scale_factor
        ax.set_yticks(y_ticks)

    # Show or hide dashed, transparent, thin gridlines
    if is_gridline:
        gridline_style = {'linestyle': 'dashed', 'alpha': 0.5, "linewidth":0.25}
        ax.xaxis._axinfo["grid"].update(gridline_style)
        ax.yaxis._axinfo["grid"].update(gridline_style)
        ax.zaxis._axinfo["grid"].update(gridline_style)
    else:
        ax.grid(False)

    # Adjust layout to prevent color bar overlapping with Y-axis
    fig.subplots_adjust(right=0.85)
    
    ax.zaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.zaxis.get_major_formatter().set_powerlimits((-3, 3))
    
    if zlim and z_interval:
        z_min, z_max = zlim
        z_ticks = np.arange(z_min, z_max + z_interval, z_interval)
        ax.set_zticks(z_ticks)

    # Show color legend if enabled
    if is_legend:
        cbar = fig.colorbar(surf, ax=ax, shrink=legend_bar_size, aspect=10, pad=0.1)
        if zlim and z_interval:
            z_min, z_max = zlim
            cbar_ticks = np.arange(z_min, z_max + z_interval, z_interval)
            cbar.ax.yaxis.set_major_locator(ticker.FixedLocator(cbar_ticks))
            cbar_labels = [str(int(t)) if t.is_integer() else '' for t in cbar_ticks]
            cbar.ax.yaxis.set_major_formatter(ticker.FixedFormatter(cbar_labels))

    # Save or display the plot
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.show()
    
def plot_grouped_bar(
    matrix, labels, group_labels,
    bar_colors=None, bar_width=0.2, bar_alpha=0.8,
    xlabel=None, ylabel=None, title=None,
    fontsize=12, tick_fontsize=10, tick_rotation=0,
    y_grid=True, grid_style="--", grid_alpha=0.5, grid_linewidth=0.5,
    y_interval=None, y_log_scale=False, y_sci_notation=False,
    figsize=(8, 6), dpi=300, save_path=None
):
    """
    Plots a grouped bar plot for a 3x3 matrix.

    Parameters:
    - matrix (numpy array or list of lists): A 3×3 matrix where rows are categories and columns are groups.
    - labels (list): Labels for each group of bars (x-axis).
    - group_labels (list): Labels for different categories (legend).
    - bar_colors (list): Colors for each group.
    - bar_width (float): Width of each bar.
    - bar_alpha (float): Transparency of bars.
    - xlabel (str): Label for x-axis.
    - ylabel (str): Label for y-axis.
    - title (str): Title of the plot.
    - fontsize (int): Font size for labels and title.
    - tick_fontsize (int): Font size for axis ticks.
    - tick_rotation (int): Rotation angle for x-axis ticks.
    - y_grid (bool): Show or hide horizontal grid lines.
    - grid_style (str): Grid line style.
    - grid_alpha (float): Transparency of grid lines.
    - grid_linewidth (float): Thickness of grid lines.
    - y_interval (float): Interval for y-axis ticks.
    - y_log_scale (bool): Use logarithmic scale on y-axis.
    - y_sci_notation (bool): Use scientific notation for y-axis.
    - figsize (tuple): Figure size.
    - dpi (int): Resolution of the figure.
    - save_path (str): Path to save the figure.

    Returns:
    - None (Displays or saves the plot).
    """

    num_groups = len(matrix)  # Number of groups (rows)
    num_bars = len(matrix[0])  # Number of bars per group (columns)
    
    x = np.arange(num_groups)  # Base x positions for each group

    if bar_colors is None:
        bar_colors = ['blue', 'orange', 'green']  # Default colors for bars

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Plot each set of bars for a group
    for i in range(num_bars):
        ax.bar(x + i * bar_width, matrix[:, i], width=bar_width, alpha=bar_alpha,
               color=bar_colors[i], label=group_labels[i])

    # Set x-axis labels and ticks
    ax.set_xticks(x + bar_width)  # Adjust tick positions to align with groups
    ax.set_xticklabels(labels, fontsize=tick_fontsize, rotation=tick_rotation)

    # Labels and title
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=fontsize)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=fontsize)
    if title:
        ax.set_title(title, fontsize=fontsize + 2)

    # Y-axis scientific notation or log scale
    if y_sci_notation:
        ax.yaxis.get_major_formatter().set_powerlimits((-3, 3))
    if y_log_scale:
        ax.set_yscale('log')

    # Set custom Y-axis tick interval
    if y_interval:
        ax.yaxis.set_major_locator(plt.MultipleLocator(y_interval))

    # Show grid
    if y_grid:
        ax.yaxis.grid(True, linestyle=grid_style, alpha=grid_alpha, linewidth=grid_linewidth)

    # Add legend
    ax.legend(fontsize=tick_fontsize)

    # Save or show the plot
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.show()
    
def plot_new_grouped_dot_chart(
    matrix, labels, group_labels,
    dot_size=100, dot_alpha=0.8, cmap='tab20', colors=None,
    xlabel=None, ylabel="Values", title=None,
    fontsize=12, tick_fontsize=10, tick_rotation=0,
    x_interval=None, y_interval=None, ylim=None, y_log_scale=False, y_sci_notation=False,
    grid_style="--", grid_alpha=0.5, grid_linewidth=0.5, is_grid=True,
    font_name='Arial', is_legend=True, is_show=True, fig_size=(10, 8), dpi=300, save_path=None
):
    """
    Plots a grouped dot chart with customizable settings.

    Parameters:
    - matrix (numpy array or list of lists): A 3×3 matrix where rows are categories and columns are groups.
    - labels (list): Labels for each group of dots (x-axis).
    - group_labels (list): Labels for different categories (legend).
    - dot_size (int): Size of dots.
    - dot_alpha (float): Transparency of dots.
    - cmap (str): Colormap for assigning colors to groups (used if colors=None).
    - colors (list): List of custom colors for each group.
    - xlabel (str): Label for x-axis.
    - ylabel (str): Label for y-axis.
    - title (str): Title of the plot.
    - fontsize (int): Font size for labels and title.
    - tick_fontsize (int): Font size for axis ticks.
    - tick_rotation (int): Rotation angle for x-axis ticks.
    - x_interval (float): Interval for x-axis ticks.
    - y_interval (float): Interval for y-axis ticks.
    - ylim (tuple): Custom Y-axis limits (min, max).
    - y_log_scale (bool): Use logarithmic scale on y-axis.
    - y_sci_notation (bool): Use scientific notation for y-axis.
    - grid_style (str): Style of the grid lines.
    - grid_alpha (float): Transparency of grid lines.
    - grid_linewidth (float): Thickness of grid lines.
    - is_grid (bool): Show or hide grid.
    - font_name (str): Font family for the plot.
    - is_legend (bool): Show or hide legend.
    - is_show (bool): Show plot.
    - fig_size (tuple): Figure size (width, height).
    - dpi (int): Resolution of the figure.
    - save_path (str): Path to save the figure.

    Returns:
    - None (Displays or saves the plot).
    """

    num_groups = len(matrix)  # Number of groups (rows)
    num_dots = len(matrix[0])  # Number of dots per group (columns)
    
    x = np.arange(num_groups)  # Base x positions for each group

    # Assign colors: use provided colors or generate from cmap
    if colors is None:
        cmap_colors = plt.get_cmap(cmap).colors  # Extract colors from colormap
        colors = [cmap_colors[i % len(cmap_colors)] for i in range(num_dots)]
    color_dict = {group_labels[i]: colors[i] for i in range(num_dots)}

    # Create the plot with custom figure size
    fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)

    # Set font properties
    plt.rcParams.update({'font.size': tick_fontsize, 'font.family': font_name})

    # Offset to differentiate the dots within the same category group
    offset = np.linspace(-0.3, 0.3, num_dots)

    for i in range(num_dots):
        # Positions for each dot
        positions = x + offset[i]
        # Plot each category
        ax.scatter(positions, matrix[:, i], color=color_dict[group_labels[i]],
                   label=group_labels[i], s=dot_size, alpha=dot_alpha)
        # Draw vertical lines
        for pos, y in zip(positions, matrix[:, i]):
            ax.vlines(pos, ymin=0, ymax=y, color=color_dict[group_labels[i]], linewidth=1, alpha=0.7)

    # Set x-ticks and labels with custom sizes
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=tick_fontsize, rotation=tick_rotation)

    # Set labels and title
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=fontsize)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=fontsize)
    if title:
        ax.set_title(title, fontsize=fontsize + 2)

    # Set Y-axis limits if provided
    if ylim:
        ax.set_ylim(ylim)

    # Set Y-axis tick formatting
    if y_sci_notation:
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        ax.yaxis.get_major_formatter().set_powerlimits((-3, 3))
    if y_log_scale:
        ax.set_yscale('log')

    # Set custom X and Y tick intervals
    if x_interval:
        ax.xaxis.set_major_locator(ticker.MultipleLocator(x_interval))
    if y_interval:
        ax.yaxis.set_major_locator(ticker.MultipleLocator(y_interval))

    # Show grid if enabled
    if is_grid:
        ax.yaxis.grid(True, linestyle=grid_style, alpha=grid_alpha, linewidth=grid_linewidth)

    # Optionally add legend
    if is_legend:
        ax.legend(title='Categories', title_fontsize=fontsize, fontsize=tick_fontsize)

    # Save or display the plot
    if save_path:
        plt.savefig(save_path, format='png', dpi=dpi, bbox_inches='tight')

    if is_show:
        plt.show()
    plt.close()
    
def plot_3d_density_surface(
    data, key_x, key_y,
    grid_size=100, cmap='viridis',
    flip_x=False, flip_y=False,
    xlabel=None, ylabel=None, zlabel='Density',
    title=None, fontsize=12, tick_fontsize=10, tick_pad=2,
    xlim=None, ylim=None, zlim=None,
    x_interval=None, y_interval=None, z_interval=None,
    x_scale_factor=1, y_scale_factor=1,
    figsize=(10, 7), dpi=300,
    save_path=None,
    is_gridline=True, background_color='white',
    legend_bar_size=0.5, is_legend=True
):
    """
    Plots a 3D density surface based on two keys from a list of dictionaries.

    Parameters:
    - data (list of dict): Input data.
    - key_x (str): Key for the x-axis values.
    - key_y (str): Key for the y-axis values.
    - grid_size (int): Resolution of the density surface (default: 100).
    - cmap (str): Colormap for the surface plot.
    - flip_x (bool): If True, reverses the x-axis.
    - flip_y (bool): If True, reverses the y-axis.
    - xlabel (str): Label for x-axis (default: key_x).
    - ylabel (str): Label for y-axis (default: key_y).
    - zlabel (str): Label for z-axis.
    - title (str): Plot title.
    - fontsize (int): Font size for labels.
    - tick_fontsize (int): Font size for axis ticks.
    - tick_pad (int): Padding between tick marks and tick labels.
    - xlim (tuple): Custom x-axis limits (min, max).
    - ylim (tuple): Custom y-axis limits (min, max).
    - zlim (tuple): Custom z-axis limits (min, max).
    - x_interval (float): Interval for x-axis ticks.
    - y_interval (float): Interval for y-axis ticks.
    - z_interval (float): Interval for z-axis and color bar ticks.
    - x_scale_factor (float): Scale factor for x-axis values.
    - y_scale_factor (float): Scale factor for y-axis values.
    - figsize (tuple): Figure size (width, height).
    - dpi (int): Resolution of the figure.
    - save_path (str): If provided, saves the figure to this path.
    - is_gridline (bool): Show or hide dashed, transparent, thin gridlines.
    - background_color (str): Background color of the plot.
    - legend_bar_size (float): Size of the color legend bar.
    - is_legend (bool): Show or hide the color legend.

    Returns:
    - None (Displays the plot or saves if a path is given).
    """

    # Extract x and y values from the list of dictionaries and apply scaling
    x = np.array([d[key_x] for d in data if key_x in d and key_y in d]) / x_scale_factor
    y = np.array([d[key_y] for d in data if key_x in d and key_y in d]) / y_scale_factor

    # Perform Kernel Density Estimation (KDE)
    xy = np.vstack([x, y])
    kde = gaussian_kde(xy)

    # Create a mesh grid for the density estimation
    x_min, x_max = (x.min(), x.max()) if xlim is None else (xlim[0] / x_scale_factor, xlim[1] / x_scale_factor)
    y_min, y_max = (y.min(), y.max()) if ylim is None else (ylim[0] / y_scale_factor, ylim[1] / y_scale_factor)
    x_grid, y_grid = np.linspace(x_min, x_max, grid_size), np.linspace(y_min, y_max, grid_size)
    X, Y = np.meshgrid(x_grid, y_grid)

    # Compute density values
    positions = np.vstack([X.ravel(), Y.ravel()])
    Z = kde(positions).reshape(grid_size, grid_size)

    # If zlim is specified, ensure the color mapping matches
    norm = None
    if zlim:
        Z = np.clip(Z, zlim[0], zlim[1])  # Clip values to zlim range
        norm = mcolors.Normalize(vmin=zlim[0], vmax=zlim[1])

    # Plot the 3D surface
    fig = plt.figure(figsize=figsize, dpi=dpi, facecolor=background_color)
    ax = fig.add_subplot(111, projection='3d')

    # Set background color
    ax.set_facecolor(background_color)

    # Plot surface with proper color normalization
    surf = ax.plot_surface(X, Y, Z, cmap=cmap, edgecolor='none', norm=norm)

    # Customizations
    ax.set_xlabel(xlabel if xlabel else '', fontsize=fontsize, labelpad=tick_pad)
    ax.set_ylabel(ylabel if ylabel else '', fontsize=fontsize, labelpad=tick_pad)
    ax.set_zlabel(zlabel, fontsize=fontsize, labelpad=tick_pad)
    if title:
        ax.set_title(title, fontsize=fontsize + 2)

    # Flip axes if needed
    if flip_x:
        ax.invert_xaxis()
    if flip_y:
        ax.invert_yaxis()

    # Set axis limits if provided
    if xlim:
        ax.set_xlim(xlim[0] / x_scale_factor, xlim[1] / x_scale_factor)
    if ylim:
        ax.set_ylim(ylim[0] / y_scale_factor, ylim[1] / y_scale_factor)
    if zlim:
        ax.set_zlim(zlim)

    # Set tick font sizes and padding
    ax.tick_params(axis='x', labelsize=tick_fontsize, pad=tick_pad)
    ax.tick_params(axis='y', labelsize=tick_fontsize, pad=tick_pad)
    ax.tick_params(axis='z', labelsize=tick_fontsize, pad=tick_pad)

    # Set custom tick intervals for X, Y, and Z axes
    if xlim and x_interval:
        x_ticks = np.arange(xlim[0], xlim[1] + x_interval, x_interval) / x_scale_factor
        ax.set_xticks(x_ticks)
    
    if ylim and y_interval:
        y_ticks = np.arange(ylim[0], ylim[1] + y_interval, y_interval) / y_scale_factor
        ax.set_yticks(y_ticks)

    # Show or hide dashed, transparent, thin gridlines
    if is_gridline:
        gridline_style = {'linestyle': 'dashed', 'alpha': 0.5, "linewidth":0.25}
        ax.xaxis._axinfo["grid"].update(gridline_style)
        ax.yaxis._axinfo["grid"].update(gridline_style)
        ax.zaxis._axinfo["grid"].update(gridline_style)
    else:
        ax.grid(False)

    # Adjust layout to prevent color bar overlapping with Y-axis
    fig.subplots_adjust(right=0.85)
    
    ax.zaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.zaxis.get_major_formatter().set_powerlimits((-3, 3))
    
    if zlim and z_interval:
        z_min, z_max = zlim
        z_ticks = np.arange(z_min, z_max + z_interval, z_interval)
        ax.set_zticks(z_ticks)

    # Show color legend if enabled
    if is_legend:
        cbar = fig.colorbar(surf, ax=ax, shrink=legend_bar_size, aspect=10, pad=0.1)
        if zlim and z_interval:
            z_min, z_max = zlim
            cbar_ticks = np.arange(z_min, z_max + z_interval, z_interval)
            cbar.ax.yaxis.set_major_locator(ticker.FixedLocator(cbar_ticks))
            cbar_labels = [str(int(t)) if t.is_integer() else '' for t in cbar_ticks]
            cbar.ax.yaxis.set_major_formatter(ticker.FixedFormatter(cbar_labels))

    # Save or display the plot
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.show()
    
def plot_yearly_trends_3d(
    df, subsets, year_col, value_col=None, start_year=None, end_year=None, 
    xlabel="Year", ylabel="Groups (Offset for Separation)", zlabel=None, title="Yearly Trends of Different Groups (3D View)", 
    label_font_size=12, tick_font_size=10, title_font_size=14, line_width=1.2, zlim=(0, 100),
    cmap="viridis", view_elev=30, view_azim=225, save_path=None, dpi=300, figsize=(10, 8)
):
    """
    Plots multiple parallel curves in a 3D view with filled areas.

    Parameters:
    - df (pd.DataFrame): DataFrame containing at least a year column.
    - subsets (dict): Dictionary where keys are group names and values are lists of row indices.
    - year_col (str): Column name representing the year.
    - value_col (str or None): Column name representing values to analyze, or None to use counts.
    - start_year (int or None): Start year for filtering.
    - end_year (int or None): End year for filtering.
    - xlabel (str): Label for the X-axis.
    - ylabel (str): Label for the Y-axis.
    - zlabel (str or None): Label for the Z-axis. Defaults to "Average Value" or "Count".
    - title (str): Plot title.
    - label_font_size (int): Font size for axis labels.
    - tick_font_size (int): Font size for axis ticks.
    - title_font_size (int): Font size for the title.
    - line_width (float): Width of the plot lines.
    - cmap (str): Color ramp for the curves (any valid matplotlib colormap).
    - view_elev (int): Elevation angle for the 3D plot.
    - view_azim (int): Azimuth angle for the 3D plot.
    - save_path (str or None): Path to save the figure. If None, it is not saved.
    - dpi (int): Resolution for saving the figure.
    """
    # Compute yearly trends for each subset
    yearly_trends = {}
    for key, indices in subsets.items():
        subset_df = df.loc[indices]
        
        if value_col:
            yearly_trend = subset_df.groupby(year_col)[value_col].mean()  # Compute mean value per year
        else:
            yearly_trend = subset_df.groupby(year_col).size()  # Count occurrences per year
        
        yearly_trends[key] = yearly_trend

    # Convert trends into a DataFrame, filling missing years with 0
    trend_df = pd.DataFrame(yearly_trends).fillna(0)

    # Ensure index (years) is numeric and sorted
    trend_df = trend_df.sort_index()
    years = trend_df.index.to_numpy()

    # Apply year filtering
    if start_year is not None:
        trend_df = trend_df[trend_df.index >= start_year]
    if end_year is not None:
        trend_df = trend_df[trend_df.index <= end_year]
    
    # Update years after filtering
    years = trend_df.index.to_numpy()
    
    # Generate distinct colors for groups
    norm = mcolors.Normalize(vmin=0, vmax=len(trend_df.columns))
    cmap = plt.get_cmap(cmap)  # Convert colormap name to colormap object

    # Create the 3D plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # Set a baseline (y-offset) for separating curves
    y_offset = 0
    y_gap = np.max(trend_df.values) * 1.5  # Adjust spacing dynamically

    # Plot each subset's trend in 3D
    for i, key in enumerate(trend_df.columns):
        values = trend_df[key].values
        color = cmap(norm(i))

        # Define X, Y, and Z values
        x_vals = years
        y_vals = np.full_like(years, y_offset)  # Y-axis offset to separate curves
        z_vals = values

        # Plot the 3D curve
        ax.plot(x_vals, y_vals, z_vals, color=color, lw=line_width, label=key)

        # Plot projections onto XY plane
        ax.plot(x_vals, y_vals, np.zeros_like(z_vals), color=color, lw=line_width, linestyle='dotted')

        # Fill between the curve and XY plane
        verts = [(x, y_offset, z) for x, z in zip(x_vals, z_vals)]  # Curve points
        verts += [(x, y_offset, 0) for x in x_vals[::-1]]  # Base points (XY plane)

        poly = Poly3DCollection([verts], color=color, alpha=0.3)  # Semi-transparent fill
        ax.add_collection3d(poly)

        # Increase y_offset for next curve
        y_offset += y_gap

    # Labels and formatting
    ax.set_xlabel(xlabel, fontsize=label_font_size)
    ax.set_ylabel(ylabel, fontsize=label_font_size)
    ax.set_zlabel(zlabel if zlabel else ("Average Value" if value_col else "Count"), fontsize=label_font_size)
    ax.set_title(title, fontsize=title_font_size)

    # Set y-ticks to group names at correct offsets
    ax.set_yticks(np.arange(0, y_offset, y_gap))
    ax.set_yticklabels(trend_df.columns, fontsize=tick_font_size)

    # Set tick font sizes
    ax.tick_params(axis='both', labelsize=tick_font_size)
    
    if zlim:
        # set the zlim
        ax.set_zlim(zlim)
        
    # Adjust view angle
    ax.view_init(elev=view_elev, azim=view_azim)

    # Show legend
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    # Save plot if required
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')

    plt.show()
    
