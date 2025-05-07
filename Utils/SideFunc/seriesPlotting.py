import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns
import copy
from scipy.optimize import curve_fit
from .dataInteraction import get_bar_data
from scipy.interpolate import griddata
from matplotlib.colors import to_rgba

def plot_histogram(wave_df, bins, output_folder, output_name, column = 'Hmax', figsize=(12, 6), color_hs='blue', color_angle='green', **kwargs):
    """
    Plots histograms of hs and angle columns of wave_df in two subplots.
    
    Parameters:
        wave_df (pd.DataFrame): The input data.
        hs_bins (int): Bin number for hs.
        angle_bins (int): Bin number for angle.
        output_folder (str): Folder path to save the figure.
        output_name (str): File name to save the figure.
        figsize (tuple): Figure window size.
        color_hs (str): Color of hs histogram.
        color_angle (str): Color of angle histogram.
    """
    
    fig, axs = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
    
    angle_hist, angle_bin_edges = np.histogram(wave_df[column], bins=bins, range=(0, max(wave_df[column])))
    angle_percentages = (angle_hist / angle_hist.sum()) * 100
    axs.bar(angle_bin_edges[:-1], angle_percentages, width=np.diff(angle_bin_edges), \
        color=color_angle, edgecolor='k', align='edge', **kwargs)
    axs.set_title('Histogram of WLF')
    axs.set_xlabel('WLF (m)')
    axs.set_ylabel('Frequency (%)')
    
    # Save the plot
    output_path = f"{output_folder}/{output_name}.png"
    plt.savefig(output_path)
    plt.close()
    
    print(f"Figure saved at {output_path}")

def plot_heatmap(wave_df0, hs_interval, angle_bin, output_folder, output_name, figsize=(10, 8), \
    vmin = 0, vmax = 8, grid_color='white', grid_thickness=1, color_ramp="YlOrRd", **kwargs):
    """
    Plots a heatmap based on hs and angle columns of wave_df.
    
    Parameters:
        wave_df (pd.DataFrame): The input data.
        hs_interval (float): Bin interval for hs.
        angle_bin (int): Bin number for angle.
        output_folder (str): Folder path to save the figure.
        output_name (str): File name to save the figure.
        figsize (tuple): Figure window size.
        grid_color (str): Grid color.
        grid_thickness (float): Grid thickness.
        color_ramp (str): Color ramp for the heatmap.
    """
    
    wave_df = copy.deepcopy(wave_df0)
    
    # Binning the hs and angle data
    hs_bins = np.arange(0, wave_df['hs'].max() + hs_interval, hs_interval)
    angle_bins = np.linspace(0, 180, angle_bin + 1)
    
    wave_df['angle_bin'] = pd.cut(wave_df['angle'], bins=angle_bins, labels=range(angle_bin), right=False)
    wave_df['hs_bin'] = pd.cut(wave_df['hs'], bins=hs_bins, labels=range(len(hs_bins)-1), right=False)

    # Creating a pivot table
    heatmap_data = pd.pivot_table(wave_df, values='hs', index=['hs_bin'], columns='angle_bin', aggfunc='count', fill_value=0)
    
    # Calculating mid-values for bins
    # hs_ticks = [(hs_bins[i] + hs_bins[i+1]) / 2 for i in range(len(hs_bins)-1)]
    # angle_ticks = [(angle_bins[i] + angle_bins[i+1]) / 2 for i in range(len(angle_bins)-1)]
    hs_ticks = [hs_bins[i] for i in range(len(hs_bins))]
    angle_ticks = [angle_bins[i] for i in range(len(angle_bins))]
    # Plotting
    plt.figure(figsize=figsize)
    sns.set(style="white")
    sns.heatmap(heatmap_data / wave_df.shape[0] * 100, cmap=color_ramp, cbar_kws={'label': 'Frequency (%)'},\
        vmin = vmin, vmax = vmax, linewidths=grid_thickness, linecolor=grid_color, **kwargs)
    
    plt.xlabel('Angle (°)')
    plt.ylabel('Hs (m)')
    plt.title('')
    
    # Setting tick labels to mid-point of bins and making sure angle labels are integers
    # plt.xticks(ticks=np.arange(len(hs_bins)-1) + 0.5, labels=[f"{h:.2f}" for h in hs_ticks], rotation=45)
    # plt.yticks(ticks=np.arange(angle_bin) + 0.5, labels=[f"{int(ang)}" for ang in angle_ticks], rotation=0)
    plt.yticks(ticks=np.arange(len(hs_bins)), labels=[f"{h:.1f}" for h in hs_ticks], rotation=0)
    plt.xticks(ticks=np.arange(len(angle_bins)), labels=[f"{int(ang)}" for ang in angle_ticks], rotation=0)
    plt.gca().invert_yaxis()
    # Save the plot
    output_path = f"{output_folder}/{output_name}.png"
    plt.savefig(output_path)
    plt.close()
    
    print(f"Figure saved at {output_path}")


def plot_hmax_hs_heatmap1(wave_df0, hs_interval, hmax_interval, output_folder, output_name, \
    figsize=(12, 8), grid_color='white', grid_thickness=1, vmin = 0, vmax= 8, cmap="YlOrRd", **kwargs):
    wave_df = copy.deepcopy(wave_df0)
    
    # Binning the hs and angle data
    hs_bins = np.arange(0, wave_df['hs'].max() + hs_interval, hs_interval)
    hmax_bins = np.arange(0, wave_df['Hmax'].max() + hmax_interval, hmax_interval)
    
    wave_df['hmax_bin'] = pd.cut(wave_df['Hmax'], bins=hmax_bins, labels=range(len(hmax_bins) -1), right=False)
    wave_df['hs_bin'] = pd.cut(wave_df['hs'], bins=hs_bins, labels=range(len(hs_bins)-1), right=False)
    # print(wave_df['hs_bin'])
    # print(wave_df['hmax_bin'])
    # Creating a pivot table
    heatmap_data = pd.pivot_table(wave_df, values='hs', index=['hs_bin'], columns='hmax_bin', aggfunc='count', fill_value=0)
    # print(heatmap_data)
    # Calculating mid-values for bins
    # hs_ticks = [(hs_bins[i] + hs_bins[i+1]) / 2 for i in range(len(hs_bins)-1)]
    # angle_ticks = [(angle_bins[i] + angle_bins[i+1]) / 2 for i in range(len(angle_bins)-1)]
    hs_ticks = [hs_bins[i] for i in range(len(hs_bins))]
    hmax_ticks = [hmax_bins[i] for i in range(len(hmax_bins))]
    # Plotting
    plt.figure(figsize=figsize)
    sns.set(style="white")
    sns.heatmap(heatmap_data / wave_df.shape[0] * 100, cmap=cmap, cbar_kws={'label': 'Frequency (%)'}, \
        linewidths=grid_thickness, vmin = vmin, vmax = vmax, linecolor=grid_color, **kwargs)
    
    plt.xlabel('Water Level Fluctuation (m)')
    plt.ylabel('Hs (m)')
    plt.title('')
    
    # Setting tick labels to mid-point of bins and making sure angle labels are integers
    # plt.xticks(ticks=np.arange(len(hs_bins)-1) + 0.5, labels=[f"{h:.2f}" for h in hs_ticks], rotation=45)
    # plt.yticks(ticks=np.arange(angle_bin) + 0.5, labels=[f"{int(ang)}" for ang in angle_ticks], rotation=0)
    plt.yticks(ticks=np.arange(len(hs_bins)), labels=[f"{h:.1f}" for h in hs_ticks], rotation=0)
    plt.xticks(ticks=np.arange(len(hmax_bins)), labels=[f"{hm:.2f}" for hm in hmax_ticks], rotation=0)
    plt.gca().invert_yaxis()
    # Save the plot
    output_path = f"{output_folder}/{output_name}.png"
    plt.savefig(output_path)
    plt.close()
    
    print(f"Figure saved at {output_path}")

def plot_scatter_hs_Hmax(df):
    # Create a new column that will store the counts of hs and Hmax pairs
    df['count'] = df.groupby(['hs', 'Hmax'])['hs'].transform('count')
    
    # Define marker types based on the angle condition
    normal_wave_marker = 'o'  # Circle for normal waves
    oblique_wave_marker = '^'  # Cross for oblique waves

    # Filter data into normal and oblique waves based on the angle range
    normal_waves = df[(df['angle'] >= 60) & (df['angle'] <= 120)]
    oblique_waves = df[(df['angle'] < 60) | (df['angle'] > 120)]

    # Create scatter plots
    # Note: The size of the marker 's' is multiplied by an arbitrary factor to ensure that it's visually significant.
    plt.scatter(normal_waves['hs'], normal_waves['Hmax'], s=np.sqrt(normal_waves['count']*20), 
                marker=normal_wave_marker, label='Normal Wave')
    plt.scatter(oblique_waves['hs'], oblique_waves['Hmax'], s=np.sqrt(oblique_waves['count']*20), 
                marker=oblique_wave_marker, label='Oblique Wave')

    # Add labels and title
    plt.xlabel('hs')
    plt.ylabel('Hmax')
    plt.title('Scatter Plot of hs vs Hmax')
    plt.legend()

    # Show the plot
    plt.show()

def plot_timeline(ax, timestamps, logical_data, true_color, true_linestyle, false_color, false_linestyle):
    # Function to plot a segment
    def plot_segment(start, end, value):
        if not value:
            pass
        else:
            color = true_color if value else false_color
            linestyle = true_linestyle if value else false_linestyle
            ax.fill_between(timestamps[start:end], 0, 1, 
                    color=color, linestyle=linestyle)
            ax.set_yticks([0, 1])
            ax.set_yticklabels([' ', 'T'])

    # Iterate through the logical data
    i = 0
    while i < len(logical_data):
        start = i
        while i + 1 < len(logical_data) and logical_data[i] == logical_data[i + 1]:
            i += 1

        end = i + 1
        
        # If there's a change in state at the next index, extend the current segment halfway into the next
        if end < len(logical_data) and logical_data[end] != logical_data[start]:
            plot_segment(start, end + 1, logical_data[start])
        else:
            plot_segment(start, end, logical_data[start])

        i += 1

def plot_data(timestamps, data_arrays, tick_interval=7, **kwargs):
    # Number of plots
    n_plots = len(data_arrays)

    # Create subplots with shared x-axis
    fig, axes = plt.subplots(n_plots, 1, sharex=True, figsize=(10, n_plots * 2))
    if n_plots == 1:
        axes = [axes]  # Ensure axes is a list even for a single plot

    # Customize each plot
    for i, ax in enumerate(axes):
        if i == 0:  # First plot is the logical array (timeline)
        #     plot_timeline(ax, timestamps, data_arrays[i], 
        #                   kwargs.get('true_color', 'green'), kwargs.get('true_linestyle', '-'),
        #                   kwargs.get('false_color', 'red'), kwargs.get('false_linestyle', '-'))
            ax.plot(timestamps, data_arrays[i], 
                    color=kwargs.get('curve_color', 'blue'), 
                    linestyle=kwargs.get('curve_linestyle', '-'))
        else:
            ax.plot(timestamps, data_arrays[i], 
                    color=kwargs.get('curve_color', 'blue'), 
                    linestyle=kwargs.get('curve_linestyle', '-'))

        # Customization: labels, titles, etc.
        if 'titles' in kwargs:
            ax.set_title(kwargs['titles'][i])
        if 'ylabels' in kwargs:
            ax.set_ylabel(kwargs['ylabels'][i])
    
    # Format x-axis to show dates with a given interval
    axes[-1].xaxis.set_major_locator(mdates.DayLocator(interval=tick_interval))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    ax0 = plt.subplot(1, 1, 1)
    # set ax0 background to transparent

    ax0.patch.set_alpha(1)
    # set axises of ax0 to invisible
    ax0.axis('off')
    # draw vertical line in ax0 
    

    v = 0
    while v < len(data_arrays[0]):
        start = v
        while v + 1 < len(data_arrays[0]) and data_arrays[0][v] == data_arrays[0][v + 1]:
            v += 1
        end = v + 1
        if data_arrays[0][start]:
            axv_ind = (start + end) / 2
            ax0.axvline(x=axv_ind, ymin=0, ymax=1, color='black', linestyle='--')
        v += 1



    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()

def plot_data_V2(timestamps, data_arrays, tick_interval=7, **kwargs):
    # Number of plots
    n_plots = len(data_arrays)

    # Create subplots with shared x-axis
    fig = plt.figure(figsize=(10, n_plots * 2))
    # Customize each plot
    for i in range(n_plots):
        ax = fig.add_subplot(n_plots, 1, i + 1)
        if i == 0:  # First plot is the logical array (timeline)
        #     plot_timeline(ax, timestamps, data_arrays[i], 
        #                   kwargs.get('true_color', 'green'), kwargs.get('true_linestyle', '-'),
        #                   kwargs.get('false_color', 'red'), kwargs.get('false_linestyle', '-'))
            ax.plot(timestamps, data_arrays[i], 
                    color=kwargs.get('curve_color', 'blue'), 
                    linestyle=kwargs.get('curve_linestyle', '-'))
        else:
            ax.plot(timestamps, data_arrays[i], 
                    color=kwargs.get('curve_color', 'blue'), 
                    linestyle=kwargs.get('curve_linestyle', '-'))

        # Customization: labels, titles, etc.
        if 'titles' in kwargs:
            ax.set_title(kwargs['titles'][i])
        if 'ylabels' in kwargs:
            ax.set_ylabel(kwargs['ylabels'][i])
        if i == n_plots - 1:
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=tick_interval))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        else:
            ax.xaxis.set_visible(False)
   
    ax0 = fig.add_subplot(1, 1, 1)
    # set ax0 background to transparent

    ax0.patch.set_alpha(1)
    # set axises of ax0 to invisible
    ax0.axis('off')
    # draw vertical line in ax0 
    
    total_time = (timestamps[-1] - timestamps[0]).total_seconds()
    v = 0
    while v < len(data_arrays[0]):
        start = v
        while v + 1 < len(data_arrays[0]) and data_arrays[0][v] == data_arrays[0][v + 1]:
            v += 1
        end = v + 1
        if data_arrays[0][start]:
            start_time = (timestamps[start] - timestamps[0]).total_seconds()
            end_time = (timestamps[end] - timestamps[0]).total_seconds()
            average_time = (start_time + end_time) / 2
            axv_ind = average_time / total_time * len(timestamps)
            print(axv_ind)
            print(timestamps[start],timestamps[end])
            #ax0.axvline(x=timestamps[round(axv_ind)], ymin=0, ymax=1, color='black', linestyle='--')
            #ax0.fill_between(timestamps[int(axv_ind):int(axv_ind)+1], 0, 1, color='black', linestyle='--')
        v += 1

    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()

def plot_center_histogram(data_dict, plot_type='both', x_bins=10, y_bins=10, xlim = [0, 4000], ylim = [0, 30], x_color='blue', y_color='red'):
    """
    Plots histograms for x and y center coordinates based on the specified plot type with frequencies as percentages.

    Args:
    data_dict (dict): Dictionary containing the timestamps and corresponding x and y center coordinates.
    plot_type (str): Type of plot to draw ('x', 'y', or 'both').
    x_bins (int): Number of bins for the x center histogram.
    y_bins (int): Number of bins for the y center histogram.
    x_color (str): Color for the x center histogram.
    y_color (str): Color for the y center histogram.
    """
    x_centers = []
    y_centers = []

    # Extract x_center and y_center values from the dictionary
    for timestamp in data_dict:
        x_centers.extend(data_dict[timestamp]['x_center'])
        y_centers.extend(data_dict[timestamp]['y_center'])

    # Calculate the weights for percentage normalization
    x_weights = [100. / len(x_centers)] * len(x_centers)
    y_weights = [100. / len(y_centers)] * len(y_centers)

    # Create histograms based on the specified plot type
    if plot_type in ['x', 'both']:
        plt.figure(figsize=(5, 3))
        plt.hist(x_centers, bins=x_bins, weights=x_weights, color=x_color)
        plt.xlabel('Alongshore Position (m)')
        plt.ylabel('Percent (%)')
        plt.xlim(xlim)
        plt.ylim(ylim)
        x_ticks = plt.xticks()[0]
        plt.xticks(x_ticks, [f"{int(x/10):d}" for x in x_ticks])
        plt.show()

    if plot_type in ['y', 'both']:
        plt.figure(figsize=(5, 3))
        plt.hist(y_centers, bins=y_bins, weights=y_weights, color=y_color)
        plt.xlabel('Offshore Position (m)')
        plt.ylabel('Percent (%)')
        plt.xlim(xlim)
        plt.ylim(ylim)
        x_ticks = plt.xticks()[0]
        plt.xticks(x_ticks, [f"{int(x/10):d}" for x in x_ticks])
        plt.show()

def plot_max_histogram(data_dict, bins = 10, color = 'blue', xlim = [0, 4000], ylim = [0, 30],):
    """
    Plots a histogram for the maximum distance from the shoreline with frequencies as percentages.

    Args:
    data_dict (dict): Dictionary containing the timestamps and corresponding maximum distances.
    bins (int): Number of bins for the histogram.
    color (str): Color for the histogram.
    """
    max_dists = []

    # Extract max_dist values from the dictionary
    for timestamp in data_dict:
        max_dists.append(data_dict[timestamp])

    # Calculate the weights for percentage normalization
    weights = [100. / len(max_dists)] * len(max_dists)

    # Create histogram
    plt.figure(figsize=(5, 3))
    plt.hist(max_dists, bins=bins, weights=weights, color=color)
    plt.xlabel('Distance from Shoreline (m)')
    plt.ylabel('Percent (%)')
    plt.xlim(xlim)
    plt.ylim(ylim)
    x_ticks = plt.xticks()[0]
    plt.xticks(x_ticks, [f"{int(x/10):d}" for x in x_ticks])
    plt.show()


def plot_3D_mesh(wave_df, hs_interval, output_folder, output_name, angle_interval = 10, figsize=(12, 8), cmap="viridis", **kwargs):
    # Create bins for hs and angle
    hs_bins = np.arange(0, wave_df['hs'].max() + hs_interval, hs_interval)
    angle_bins = np.arange(0, 180, angle_interval)  # fixed 10-degree bins for angles

    # Calculate histogram
    H, xedges, yedges = np.histogram2d(wave_df['hs'], wave_df['angle'], bins=[hs_bins, angle_bins])
    

    # Set up the figure and 3D axis
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # Create a meshgrid for the 3D surface plot
    x, y = np.meshgrid((xedges[:-1] + xedges[1:]) / 2, (yedges[:-1] + yedges[1:]) / 2)

    # Plot the surface
    ax.plot_surface(x, y, H.T, cmap=cmap, **kwargs)  # H needs to be transposed for correct orientation

    ax.set_xlabel('Hs')
    ax.set_ylabel('Angle')
    ax.set_zlabel('Frequency')

    # Save the plot
    output_path = f"{output_folder}/{output_name}.png"
    plt.savefig(output_path)
    plt.close()
    
    print(f"Figure saved at {output_path}")

def custom_plot(arr_1, arr_2, folder, name, interval = 5, line_type='-', line_color='blue', line_thickness=1, point_size=28, point_face_color='red', point_edge_color='black', \
                figsize=(12, 8), point_type='o', fit_type='linear', ylim = [70, 100], poly_degree=2):
    
    # Define the function for linear fitting
    def linear_func(x, a, b):
        return a * x + b

    # Define the function for polynomial fitting
    def poly_func(x, *params):
        return sum(p * x**i for i, p in enumerate(params))

    # Define the function for exponential fitting
    def exp_func(x, a, b, c):
        return a * np.exp(b * x) + c
    
    def log_func(x, a, b, c):
        return a + b * np.log(x + c + 1) 
    plt.figure(figsize=figsize)
    # Create the scatter plot with the specified parameters
    plt.scatter(arr_1, arr_2, s=point_size, facecolors=point_face_color, edgecolors=point_edge_color, marker=point_type)

    # Choose the fit type and perform the fitting
    if fit_type == 'linear':
        params, params_covariance = curve_fit(linear_func, arr_1, arr_2)
        fit_func = linear_func
    elif fit_type == 'polynomial':
        params = np.polyfit(arr_1, arr_2, poly_degree)
        fit_func = lambda x, *params: sum(p * x**i for i, p in enumerate(params))
    elif fit_type == 'exponential':
        params, params_covariance = curve_fit(exp_func, arr_1, arr_2)
        fit_func = exp_func
    elif fit_type == 'logarithmic':
        params, params_covariance = curve_fit(log_func, arr_1, arr_2, maxfev=1000000)
        fit_func = log_func
    else:
        raise ValueError("Unsupported fit type: {}".format(fit_type))

    # Generate a range of x values for plotting the fit
    x_fit = np.linspace(min(arr_1), max(arr_1), 1000)
    plt.xticks(list(range(0, len(arr_1) + 1, interval)))
    plt.ylim([ylim[0], ylim[1]])
    # Plot the fit with the specified parameters
    plt.plot(x_fit, fit_func(x_fit, *params), line_type, color=line_color, linewidth=line_thickness)
    plt.gca().tick_params(colors='black', labelsize = 18)
    # Save the plot to the specified folder with the specified name
    plt.savefig(f'{folder}/{name}.png')
    #plt.gca().legend(('a','b'))
    # Display the plot
    plt.show()

def custom_barplot(arr_1, arr_2, folder, name, bar_face_color='blue', bar_edge_color='black', bar_width=0.4, text_color='black', border_color='black', text=None, text_position=(0.1, 0.9)):
    # Create the bar plot with specified parameters
    plt.bar(arr_1, arr_2, color=bar_face_color, edgecolor=bar_edge_color, width=bar_width)

    # Set the border colors
    for spine in plt.gca().spines.values():
        spine.set_edgecolor(border_color)

    # Set text color for labels, title, and ticks
    plt.gca().tick_params(colors=text_color)
    plt.xlabel('X-axis', color=text_color)
    plt.ylabel('Y-axis', color=text_color)
    plt.title('Custom Bar Plot', color=text_color)
    
    # Add text annotation
    if text:
        plt.text(text_position[0], text_position[1], text, transform=plt.gca().transAxes, color=text_color)

    # Save the plot to the specified folder with the specified name
    plt.savefig(f'{folder}/{name}.png')

    # Display the plot
    plt.show()

def plot_3D_bars(wave_df, hs_interval, output_folder, output_name, angle_interval = 15, \
                figsize=(12, 8), cmap="viridis", shaded = False, **kwargs):
    # Create bins for hs and angle
    hs_bins = np.arange(0, wave_df['hs'].max() + hs_interval, hs_interval)
    angle_bins = np.arange(0, 181, angle_interval)  # fixed 10-degree bins for angles

    # Calculate histogram
    H, xedges, yedges = np.histogram2d(wave_df['hs'], wave_df['angle'], bins=[hs_bins, angle_bins])

    # Set up the figure and 3D axis
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # Create 3D bars
    dx = hs_interval  # width of bars along the x-axis (Hs)
    dy = angle_interval  # width of bars along the y-axis (Angle)
    max_frequency = H.max()
    colormap = plt.get_cmap(cmap)
    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            color = colormap(H[i, j] / max_frequency)  # Normalize value to [0, 1] range
            ax.bar3d(xedges[i], yedges[j], 0, dx, dy, H[i, j], shade=shaded, color=color)

    # dx = hs_interval  # width of bars along x-axis (Hs)
    # dy = 10  # width of bars along y-axis (Angle)
    # for i in range(H.shape[0]):
    #     for j in range(H.shape[1]):
    #         ax.bar(left=xedges[i], height=H[i,j], depth=dy, bottom=yedges[j], zdir='y', width=dx, shade=True, color=cmap(H[i,j]))

    ax.set_xlabel('Hs')
    ax.set_ylabel('Angle')
    ax.set_zlabel('Frequency')

    # Save the plot
    output_path = f"{output_folder}/{output_name}.png"
    plt.savefig(output_path)
    plt.close()
    
    print(f"Figure saved at {output_path}")

def plot_scatter_hs_Hmax(df):
    # Create a new column that will store the counts of hs and Hmax pairs
    df['count'] = df.groupby(['hs', 'Hmax'])['hs'].transform('count')
    
    # Define marker types based on the angle condition
    normal_wave_marker = 'o'  # Circle for normal waves
    oblique_wave_marker = '^'  # Cross for oblique waves

    # Filter data into normal and oblique waves based on the angle range
    normal_waves = df[(df['angle'] >= 60) & (df['angle'] <= 120)]
    oblique_waves = df[(df['angle'] < 60) | (df['angle'] > 120)]

    # Create scatter plots
    # Note: The size of the marker 's' is multiplied by an arbitrary factor to ensure that it's visually significant.
    plt.scatter(normal_waves['hs'], normal_waves['Hmax'], s=np.sqrt(normal_waves['count']*20), 
                marker=normal_wave_marker, label='Normal Wave')
    plt.scatter(oblique_waves['hs'], oblique_waves['Hmax'], s=np.sqrt(oblique_waves['count']*20), 
                marker=oblique_wave_marker, label='Oblique Wave')

    # Add labels and title
    plt.xlabel('hs')
    plt.ylabel('Hmax')
    plt.title('Scatter Plot of hs vs Hmax')
    plt.legend()

    # Show the plot
    plt.show()

def plot_histograms(list_of_lists, titles, xlabels='', ylabels='', bins=10, color='blue', edgecolor='black', fig_size=(10, 6), tick_fontname='Arial',
                    tick_fontsize=12, title_fontsize=14, label_fontsize=14, value_range=None, line_color='red', show_all_xticklabels=True,
                    line_style='--', line_width=2, is_legend=False, unit='m', is_log=False, is_log_x=False, is_fixed_y_range=False, 
                    y_range=None, is_mean_value = True,
                    is_scaled=False, scale_factor=10, save_path='', is_show=False, is_save=True, transparent_bg=True, hspace=0.1):
    
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
            axs[i].set_title(titles[i], fontsize=title_fontsize, fontname=tick_fontname)
        
        if xlabels[i]:
            axs[i].set_xlabel(xlabels[i], fontsize=label_fontsize, fontname=tick_fontname)
        else:
            axs[i].set_xticks([])
        
        if ylabels[i]:
            axs[i].set_ylabel(ylabels[i], fontsize=label_fontsize, fontname=tick_fontname)
        
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

    plt.tight_layout()
    plt.subplots_adjust(hspace=hspace)
    if is_show:
        plt.show()
    if is_save:
        if save_path:
            fig.savefig(save_path, dpi=600, transparent=transparent_bg)
        else:
            print("Please provide a valid path to save the figure.")



def draw_barplot(result, bar_color='blue', bar_thickness=0.8, bar_edge_color='black',
                 value_range=None, figsize=(10, 6), tick_fontsize=10, tick_fontname='Arial'):
    
    dates = list(result.keys())
    values = list(result.values())
    
    plt.figure(figsize=figsize)
    bars = plt.bar(dates, values, color=bar_color, edgecolor=bar_edge_color, width=bar_thickness)
    
    if value_range:
        plt.ylim(value_range)
    
    plt.xticks(fontsize=tick_fontsize, fontname=tick_fontname)
    plt.yticks(fontsize=tick_fontsize, fontname=tick_fontname)
    
    plt.xlabel('Date', fontsize=tick_fontsize + 2, fontname=tick_fontname)
    plt.ylabel('Value', fontsize=tick_fontsize + 2, fontname=tick_fontname)
    plt.title('Bar Plot of Computation Results', fontsize=tick_fontsize + 4, fontname=tick_fontname)
    
    plt.show()

def draw_multibarplots(main_result, other_data_list, bar_color='blue', bar_thickness=0.8, bar_edge_color='black', line_color = 'black', 
                       y_range=None, figsize=(10, 6), line_thickness = 1, tick_fontsize=10, tick_fontname='sans-serif', x_tick_interval=1, is_show = False, 
                       is_save = True, save_path = ''):
    def prepare_data(result):
        dates = list(result.keys())
        values = list(result.values())
        return pd.Series(values, index=pd.to_datetime(dates))
    def is_number(variable):
        return isinstance(variable, (int, float))

    main_series = prepare_data(main_result)
    all_series = [prepare_data(data) for data in other_data_list]

    fig, axes = plt.subplots(len(all_series) + 1, 1, figsize=figsize, sharex=True)
    
    # Plot the main result as a bar plot
    axes[0].bar(main_series.index, main_series.values, color=bar_color, edgecolor=bar_edge_color, width=bar_thickness)
    axes[0].tick_params(axis='x', labelsize=tick_fontsize)
    axes[0].tick_params(axis='y', labelsize=tick_fontsize)
    for tick in axes[0].get_xticklabels():
        tick.set_fontname(tick_fontname)
    for tick in axes[0].get_yticklabels():
        tick.set_fontname(tick_fontname)
    if y_range:
        axes[0].set_ylim(y_range[0])

    # Plot each additional dataset as a line plot
    for idx, series in enumerate(all_series, start=1):
        axes[idx].plot(series.index, series.values, color=line_color)
        axes[idx].tick_params(axis='x', labelsize=tick_fontsize)
        axes[idx].tick_params(axis='y', labelsize=tick_fontsize)
        for tick in axes[idx].get_xticklabels():
            tick.set_fontname(tick_fontname)
        for tick in axes[idx].get_yticklabels():
            tick.set_fontname(tick_fontname)
        if y_range:
            axes[idx].set_ylim(y_range[idx])
        if line_thickness != None:
            if is_number(line_thickness):
                axes[idx].plot(series.index, series.values, color=line_color, linewidth=line_thickness)
            else:
                axes[idx].plot(series.index, series.values, color=line_color, linewidth=line_thickness[idx - 1])
                

    # Set date format on x-axis and set tick interval for all subplots
    axes[-1].xaxis.set_major_locator(mdates.DayLocator(interval=x_tick_interval))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    fig.autofmt_xdate()  # Auto format x-axis dates for better appearance

    fig.tight_layout()
    if is_show:
        plt.show()
    if is_save:
        if save_path:
            fig.savefig(save_path, dpi=600)
        else:
            print("Please provide a valid path to save the figure.")

def plot_bar_plots(list_of_lists, tuple_range_list, titles = '', ylabels='', bar_color='blue', bar_edgecolor='black', fig_size=(10, 6), tick_fontname='Arial',
                    tick_fontsize=12, title_fontsize=14, label_fontsize=14, line_color='red', show_all_xticklabels=True, bar_width = 1,
                    line_style='--', line_width=2, is_legend=False, unit='m', is_fixed_y_range=True, y_range=[0, 20], is_mean_value = False,
                    is_scaled=False, scale_factor=10, save_path='', is_show=False, is_save=True, transparent_bg=True, horizontal = True, 
                    convert_minute = True, hspace=0.05):
    n = len(list_of_lists)
    w, h = fig_size
    
    if is_fixed_y_range and y_range is None:
        max_bar_value = 0
        for data in list_of_lists:
            if is_scaled:
                data = np.array(data) / scale_factor
            bars = get_bar_data(data, tuple_range_list)
            max_bar_value = max(max_bar_value, bars.max())
        y_range = [0, max_bar_value * 1.05]
    
    fig, axs = plt.subplots(n, 1, figsize=(w, h * n))

    for i, data in enumerate(list_of_lists):
        if is_scaled:
            data = np.array(data) / scale_factor
        
        bar_positions = np.arange(len(tuple_range_list))
        bars = get_bar_data(data, tuple_range_list)  # This function needs to be defined to get bar data
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

def draw_multibarplots_with_category(main_result, other_data_list, bar_colors=None, bar_thickness=0.8, bar_edge_color='black', line_color='black', 
                       y_range=None, figsize=(10, 6), line_thickness=1, tick_fontsize=10, tick_fontname='sans-serif', x_tick_interval=1, is_show=False, 
                       is_save=True, save_path=''):
    def prepare_data(result):
        dates = list(result.keys())
        values = list(result.values())
        return pd.DataFrame(values, index=pd.to_datetime(dates))
    
    def is_number(variable):
        return isinstance(variable, (int, float))

    main_df = prepare_data(main_result)
    all_series = [prepare_data(data) for data in other_data_list]

    fig, axes = plt.subplots(len(all_series) + 1, 1, figsize=figsize, sharex=True)

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
    axes[0].legend()

    # Plot each additional dataset as a line plot
    for idx, series in enumerate(all_series, start=1):
        axes[idx].plot(series.index, series.values, color=line_color)
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
                axes[idx].plot(series.index, series.values, color=line_color, linewidth=line_thickness[idx - 1])

    # Set date format on x-axis and set tick interval for all subplots
    axes[-1].xaxis.set_major_locator(mdates.DayLocator(interval=x_tick_interval))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    fig.autofmt_xdate()  # Auto format x-axis dates for better appearance

    fig.tight_layout()
    if is_show:
        plt.show()
    if is_save:
        if save_path:
            fig.savefig(save_path, dpi=600)
        else:
            print("Please provide a valid path to save the figure.")

def draw_multibarplots_with_category_V2(main_result, other_data_list, missing_list = None, non_missing_list = None, bar_colors=None, bar_thickness=0.8, cat_labels = None,
                                     bar_edge_color='black', line_color='black', y_range=None, fig_size=(10, 6), 
                                     line_thickness=1, tick_fontsize=10, tick_fontname='sans-serif', 
                                     x_tick_interval=1, is_show=False, is_save=True, save_path=''):
    def prepare_data(result):
        dates = list(result.keys())
        values = list(result.values())
        return pd.DataFrame(values, index=pd.to_datetime(dates))

    def adjust_saturation(color, alpha=0.5):
        """Adjust the saturation of a color by modifying its alpha channel."""
        rgba = to_rgba(color)
        return (rgba[0], rgba[1], rgba[2], alpha)   
    
    def is_number(variable):
        return isinstance(variable, (int, float))

    main_df = prepare_data(main_result)
    all_series = [prepare_data(data) for data in other_data_list]

    fig, axes = plt.subplots(len(all_series) + 1, 1, figsize=fig_size, sharex=True)

    if bar_colors is None:
        bar_colors = ['#377eb8', '#ff7f00', '#4daf4a', '#e41a1c', '#984ea3']

    # Plot the main result as a stacked bar plot
    bottom_series = None
    for i, col in enumerate(main_df.columns):
        color = bar_colors[i % len(bar_colors)]
        axes[0].bar(main_df.index, main_df[col], bottom=bottom_series, color=color, edgecolor=bar_edge_color, 
                    width=bar_thickness, label=col)
        if bottom_series is None:
            bottom_series = main_df[col]
        else:
            bottom_series += main_df[col]

    axes[0].tick_params(axis='x', labelsize=tick_fontsize)
    axes[0].tick_params(axis='y', labelsize=tick_fontsize)
    if y_range:
        axes[0].set_ylim(y_range[0])
    if cat_labels == None:
        axes[0].legend()
    else:
        axes[0].legend(cat_labels)

    # Plot each additional dataset with saturation adjustment based on missing_ranges
    for idx, series in enumerate(all_series, start=1):
        if line_thickness is not None:
            if is_number(line_thickness):
                line_width = line_thickness
            else:
                line_width = line_thickness[idx - 1]
        if missing_list == None or non_missing_list == None:
            if line_width != 0:
                axes[idx].plot(series.index, series.values, color=line_color, linewidth = line_width)
            else:
                axes[idx].scatter(series.index, series.values, color='black', s=1)
        else:
            for missing_start, missing_end in missing_list:
                mask = (series.index >= missing_start) & (series.index <= missing_end)
                axes[idx].plot(series.index[mask], series[mask].values, color=adjust_saturation('black', 0.5), linewidth = line_width)
            
            for non_missing_start, non_missing_end in non_missing_list:
                mask = (series.index >= non_missing_start) & (series.index <= non_missing_end)
                axes[idx].plot(series.index[mask], series[mask].values, color='black', linewidth = line_width)

            # non_missing_mask = ~(series.index.to_series().apply(lambda x: any(missing_start <= x <= missing_end for missing_start, missing_end in missing_list)))
            # axes[idx].plot(series.index[non_missing_mask], series[non_missing_mask].values, color=color)

        axes[idx].tick_params(axis='x', labelsize=tick_fontsize)
        axes[idx].tick_params(axis='y', labelsize=tick_fontsize)
        for tick in axes[idx].get_xticklabels():
            tick.set_fontname(tick_fontname)
        for tick in axes[idx].get_yticklabels():
            tick.set_fontname(tick_fontname)
        if y_range:
            axes[idx].set_ylim(y_range[idx])
        
    # Set date format on x-axis and set tick interval for all subplots
    axes[-1].xaxis.set_major_locator(mdates.DayLocator(interval=x_tick_interval))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    fig.autofmt_xdate()  # Auto format x-axis dates for better appearance

    fig.tight_layout()
    if is_show:
        plt.show()
    if is_save:
        if save_path:
            fig.savefig(save_path, dpi=600)
        else:
            print("Please provide a valid path to save the figure.")

def plot_time_histogram(histogram, color='blue', edgecolor='black', fig_size=(10, 6),
                        tick_fontname='Arial', tick_fontsize=12, title_fontsize=14, 
                        label_fontsize=14, y_range=None, line_color='red', 
                        show_all_xticklabels=True, x_ticklabel_interval=30, 
                        x_ticklabel_format='HH:MM', is_legend=False, save_path='', 
                        is_show=False, is_save=True, transparent_bg=True):
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=fig_size)
    
    # Prepare data
    dates = list(histogram.keys())
    values = list(histogram.values())
    
    # Plotting the bar chart
    ax.bar(dates, values, color=color, edgecolor=edgecolor)
    
    # Set y-axis limits if specified
    if y_range:
        ax.set_ylim(y_range)
    
    # Set x-ticks format
    if x_ticklabel_format.lower() == 'hh:mm':
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    elif x_ticklabel_format.lower() == 'yyyy-mm-dd':
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    
    # Set the interval for x-tick labels
    if show_all_xticklabels:
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=x_ticklabel_interval))
    else:
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    
    # Customize tick labels
    ax.tick_params(axis='both', labelsize=tick_fontsize, labelcolor='black')
    plt.xticks(fontname=tick_fontname)
    plt.yticks(fontname=tick_fontname)
    
    # Set title and labels
    ax.set_title('Histogram of Time Ranges', fontsize=title_fontsize)
    ax.set_xlabel('Date', fontsize=label_fontsize)
    ax.set_ylabel('Frequency', fontsize=label_fontsize)
    
    # Legend
    if is_legend:
        ax.legend()
    
    # Save plot
    if is_save:
        if save_path:
            plt.savefig(save_path, transparent=transparent_bg, bbox_inches='tight', dpi=600)
    
    # Show plot
    if is_show:
        plt.show()
    
    # Close plot to free memory
    plt.close()

def generate_ticklabels(tick_bins, is_close_1, is_close_2):
        res = []
        if not is_close_1:
            res.append(f'<{tick_bins[0]}')
        for i in range(len(tick_bins) - 1):
            res.append(f'{tick_bins[i]}-{tick_bins[i + 1]}')
        if not is_close_2:
            res.append(f'>{tick_bins[-1]}')
        return res

def plot_2D_heatmap(pair_frequency, x_bin_ticks, y_bin_ticks, fig_size=(10, 8), title='Cross Relationship Heatmap', title_fontsize=16,
                 xlabel='Variable 1', ylabel='Variable 2', label_fontsize=14, tick_fontsize=12, vmin = None, vmax = None,
                 cmap='viridis', cbar_label='Frequency', save_path='', is_show=True, x_ticklabel_left_close = False, x_ticklabel_right_close = False,
                 y_ticklabel_top_close = False, y_ticklabel_bottom_close = False,is_annotate = False, is_percent = False,
                 xtick_rotation=0, ytick_rotation=0, xticklabels=None, yticklabels=None):
    """
    Plots a heatmap of the frequency of tuple pairs.

    :param pair_frequency: Dictionary with keys as tuple pairs of values and values as their frequency.
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
    :param xtick_rotation: Rotation angle of x-tick labels.
    :param ytick_rotation: Rotation angle of y-tick labels.
    :param xticklabels: Custom labels for the x-axis ticks.
    :param yticklabels: Custom labels for the y-axis ticks.
    """
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
    # print(index)
    # print(columns)
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
    plt.title(title, fontsize=title_fontsize)
    plt.xlabel(xlabel, fontsize=label_fontsize)
    plt.ylabel(ylabel, fontsize=label_fontsize)
    
    
    # Custom or default tick labels
    if xticklabels is None:
        xticklabels = generate_ticklabels(x_bin_ticks, x_ticklabel_left_close, x_ticklabel_right_close)
        
    plt.xticks(ticks=np.arange(len(index)) + 0.5, labels=xticklabels, rotation=xtick_rotation, fontsize=tick_fontsize)
    if yticklabels is None:
        yticklabels = generate_ticklabels(y_bin_ticks, y_ticklabel_top_close, y_ticklabel_bottom_close)
        
    plt.yticks(ticks=np.arange(len(columns)) + 0.5, labels=yticklabels, rotation=ytick_rotation, fontsize=tick_fontsize)
    # Save the plot
    if save_path:
        plt.savefig(save_path, bbox_inches='tight',dpi=600)
    
    # Show the plot
    if is_show:
        plt.show()
    plt.close()


def plot_3d_stacked_bar(triple_frequency, fig_size=(10, 8), title='3D Stacked Bar Chart', title_fontsize=16,
                        xlabel='Variable 1', ylabel='Variable 2', zlabel='Variable 3', label_fontsize=14,
                        tick_fontsize=12, bar_width=0.5, color_map='viridis', save_path='', is_show=True):
    """
    Plots a 3D stacked bar chart for triple frequency data.

    :param triple_frequency: Dictionary with keys as tuple triples of values and values as their frequency.
    :param fig_size: Size of the figure.
    :param title: Title of the chart.
    :param title_fontsize: Font size for the title.
    :param xlabel, ylabel, zlabel: Labels for the axes.
    :param label_fontsize: Font size for labels.
    :param tick_fontsize: Font size for tick labels.
    :param bar_width: Width of each bar in the 3D plot.
    :param color_map: Color map of the bars.
    :param save_path: Path to save the figure.
    :param is_show: Whether to display the plot.
    """
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111, projection='3d')

    # Extract data for the bar plot
    labels = list(set((key[0], key[1]) for key in triple_frequency.keys()))
    labels.sort()  # Sorting for consistent plotting

    # Variables to hold the unique categories
    xs = np.array([label[0] for label in labels])
    ys = np.array([label[1] for label in labels])

    # Prepare data
    all_categories = sorted(set(key[2] for key in triple_frequency.keys()))
    bottom = np.zeros(len(labels))

    colors = plt.get_cmap(color_map)(np.linspace(0, 1, len(all_categories)))

    for idx, category in enumerate(all_categories):
        zs = np.array([triple_frequency.get((x, y, category), 0) for x, y in labels])
        ax.bar3d(xs, ys, bottom, bar_width, bar_width, zs, color=colors[idx], label=str(category))
        bottom += zs  # Update the bottom to stack the next category

    # Labeling
    ax.set_title(title, fontsize=title_fontsize)
    ax.set_xlabel(xlabel, fontsize=label_fontsize)
    ax.set_ylabel(ylabel, fontsize=label_fontsize)
    ax.set_zlabel(zlabel, fontsize=label_fontsize)
    
    ax.set_xticks(range(len(xs)))
    ax.set_xticklabels(xs, fontsize=tick_fontsize)
    ax.set_yticks(range(len(ys)))
    ax.set_yticklabels(ys, fontsize=tick_fontsize)
    
    # Add legend
    ax.legend(title=zlabel, fontsize=tick_fontsize)

    # Save the plot
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    # Show the plot
    if is_show:
        plt.show()
    plt.close()

def plot_contour_map(data, fig_size=(10, 8), title='Contour Map', title_fontsize=16,
                     xlabel='X Coordinate', ylabel='Y Coordinate', label_fontsize=14,
                     tick_fontsize=12, contour_levels=50, cmap='viridis', save_path='', is_show=True):
    """
    Plots a contour map for triple frequency data.

    :param data: Dictionary with keys as tuple triples (x, y, z-value) and values as frequencies.
    :param fig_size: Size of the figure.
    :param title: Title of the chart.
    :param title_fontsize: Font size for the title.
    :param xlabel, ylabel: Labels for the axes.
    :param label_fontsize: Font size for labels.
    :param tick_fontsize: Font size for tick labels.
    :param contour_levels: Number of contour levels to plot.
    :param cmap: Color map of the contour.
    :param save_path: Path to save the figure.
    :param is_show: Whether to display the plot.
    """
    # Extract data points
    points = np.array([(key[0], key[1]) for key in data.keys()])
    values = np.array(list(data.values()))

    # Generate a grid to interpolate onto
    x = np.linspace(np.min(points[:, 0]), np.max(points[:, 0]), 100)
    y = np.linspace(np.min(points[:, 1]), np.max(points[:, 1]), 100)
    X, Y = np.meshgrid(x, y)

    # Interpolate data
    Z = griddata(points, values, (X, Y), method='cubic')

    # Plotting
    plt.figure(figsize=fig_size)
    plt.contourf(X, Y, Z, levels=contour_levels, cmap=cmap)
    plt.colorbar(label='Frequency')
    plt.title(title, fontsize=title_fontsize)
    plt.xlabel(xlabel, fontsize=label_fontsize)
    plt.ylabel(ylabel, fontsize=label_fontsize)
    plt.xticks(fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)

    # Save the plot
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    # Show the plot
    if is_show:
        plt.show()
    plt.close()

def plot_polylines(df, x, ys, line_styles=None, line_widths=None, line_colors=None, legends=None, show_legend=True,
                   marker_colors=None, figsize=(10, 6), x_tick_interval=1, markers=None, y_label = None,
                   show_grid=True, font_name='Arial', font_size=12, save_path=None, dpi=600, y_range = None):
    """
    Plots multiple lines from a DataFrame using column indices for x and ys with customizable font settings
    and an option to save the figure.

    Args:
    df (DataFrame): The DataFrame containing the data.
    x (int): Index of the column to use as x-axis.
    ys (list of int): List of indices of columns to plot on the y-axis.
    line_styles (dict): Dictionary mapping column indices to line styles.
    line_widths (dict): Dictionary mapping column indices to line widths.
    line_colors (dict): Dictionary mapping column indices to line colors.
    legends (list): Optional list of legend labels.
    marker_colors (dict): Dictionary mapping column indices to marker colors.
    figsize (tuple): Figure size.
    x_tick_interval (int): Interval between x-ticks.
    markers (dict): Dictionary mapping column indices to markers.
    show_grid (bool): Whether to show grid.
    font_name (str): Font name for all text elements.
    font_size (int): Font size for all text elements.
    save_path (str): Path to save the figure. If None, the figure is not saved.
    dpi (int): The resolution in dots per inch of the saved figure.
    """
    plt.figure(figsize=figsize)
    ax = plt.gca()

    # Set global font properties
    plt.rcParams.update({'font.size': font_size, 'font.family': font_name})

    for y in ys:
        plt.plot(df.iloc[:, x], df.iloc[:, y],
                 linestyle=line_styles.get(y, '-'),  # Default to solid line
                 linewidth=line_widths.get(y, 2),    # Default line width
                 color=line_colors.get(y, 'blue'),   # Default line color
                 marker=markers.get(y, ''),          # Default no markers
                 markerfacecolor=marker_colors.get(y, 'blue'))  # Default marker color

    # Set x-ticks interval
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.xticks(rotation=0)
    plt.xlabel(df.columns[x])
    y_label = "Percent (%)" if y_label is None else y_label
    plt.ylabel(y_label)
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
    plt.show()