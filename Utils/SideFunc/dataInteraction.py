import copy
import pandas as pd
import os
from ..DateTime.BasicDateTime import time_difference, find_closest_timestamps
from ..Wave.wave_processing import weighted_average_angle
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np
from tqdm import tqdm

# this function works for splitting the result dictionary into three dictionaries based on the type dictionary
def split_result_dict(result_dict, type_dict):
    As, Bs, Cs = {}, {}, {}
    for k in result_dict.keys():
        if type_dict.get(k[:10]) == 'A' or type_dict.get(k[:10]) == 'D':
            As[k] = result_dict[k]
        elif type_dict.get(k[:10]) == 'B':
            Bs[k] = result_dict[k]
        elif type_dict.get(k[:10]) == 'C' or type_dict.get(k[:10]) == 'D':
            Cs[k] = result_dict[k]

    return As, Bs, Cs

def interpolate_values(old_df1, old_df2, column_name):
    df1 = copy.deepcopy(old_df1)
    df2 = copy.deepcopy(old_df2)

    # Ensure the index is in datetime format
    df1.index = pd.to_datetime(df1.index)
    df2.index = pd.to_datetime(df2.index)
    
    # Initialize a list to store interpolated values
    interpolated_values = []
    
    # Loop through each timestamp in df1
    for timestamp in df1.index:
        # Find the closest timestamps in df2 before and after the current timestamp
        before = df2.index[df2.index <= timestamp].max()
        after = df2.index[df2.index >= timestamp].min()
        
        # Perform linear interpolation if before and after are different
        if before != after:
            val_before = df2.loc[before]
            val_after = df2.loc[after]
            # Linear interpolation formula
            total_secs = (after - before).total_seconds()
            elapsed_secs = (timestamp - before).total_seconds()
            interpolated_value = val_before + (val_after - val_before) * (elapsed_secs / total_secs)
        else:
            # If before and after are the same, use the value directly
            interpolated_value = df2.loc[before]
        
        interpolated_values.append(interpolated_value)

    return interpolated_values

def get_subdictionary_keys(data):
    # Get the first sub-dictionary from the dictionary of dictionaries
    first_subdictionary = next(iter(data.values()))
    
    # Return the keys of the first sub-dictionary
    return list(first_subdictionary.keys())

def interpolate_vals(val_1, val_2, weight):
    return val_1 * weight + val_2 * (1 - weight)

def compute_weight(timestamp, lower_bound, upper_bound):
    time_diff_1 = abs(time_difference(timestamp, lower_bound))
    time_diff_2 = abs(time_difference(timestamp, upper_bound))
    total_time_diff = time_diff_1 + time_diff_2
    weight = time_diff_2 / total_time_diff
    return weight

def create_hist_data(whole_merged_events, col_name):
    return [whole_merged_events[i][col_name] for i in range(len(whole_merged_events))]

def process_data_within_range(dicts, start_date, end_date, check_key, method):
    """
    Process data based on the given parameters. dicts is a list of dictionaries, each dict use timestamp as the key
    and the value is the data. start_date and end_date are the date range to process. check_key is the key in data 
    (which is the subdict) to check in
    """

    def parse_datetime(dt_str):
        return datetime.strptime(dt_str, '%Y-%m-%d-%H-%M-%S')
    
    def parse_date(date_str):
        return datetime.strptime(date_str, '%Y-%m-%d').date()
    
    start_date = parse_date(start_date)
    end_date = parse_date(end_date)
    
    filtered_data = defaultdict(list)
    
    for item in dicts:
        start_time = parse_datetime(item['start_time']).date()
        end_time = parse_datetime(item['end_time']).date()
        
        if start_time <= end_date and end_time >= start_date:
            current_date = max(start_time, start_date)
            while current_date <= min(end_time, end_date):
                if check_key:
                    if check_key in item:
                        filtered_data[current_date].append(item[check_key])
                else:
                    filtered_data[current_date].append(1)
                current_date += timedelta(days=1)
    
    result = {}
    for date, values in filtered_data.items():
        if method == 'sum':
            result[date] = sum(values)
        elif method == 'mean':
            result[date] = sum(values) / len(values) if values else 0
        elif method == 'middle':
            values.sort()
            mid_index = len(values) // 2
            if len(values) % 2 == 0:
                result[date] = (values[mid_index - 1] + values[mid_index]) / 2
            else:
                result[date] = values[mid_index]
        elif method == 'max':
            result[date] = max(values)
    
    return result

def resample_data(input_dict, original_interval, new_interval):
    # this function resamples the input dictionary by taking every nth item (timestamps)
    # Calculate the ratio of the original interval to the new interval
    ratio = new_interval // original_interval
    
    # Sort the dictionary by the keys (timestamps)
    sorted_items = sorted(input_dict.items(), key=lambda x: datetime.strptime(x[0], '%Y-%m-%d-%H-%M-%S'))
    
    # Resample the data by taking every nth item
    resampled_dict = {}
    for i in range(0, len(sorted_items), ratio):
        timestamp_str, sub_dict = sorted_items[i]
        
        resampled_dict[timestamp_str] = sub_dict
    
    return resampled_dict

def print_log_for_hist_data(hist_list, hist_names, save_folder, save_file, is_scale = True, scale_factor = 10):
    os.makedirs(save_folder, exist_ok = True)
    with open(save_folder + save_file, 'w') as f:
        for i in range(len(hist_list)):
            temp_hist_data = hist_list[i]
            temp_name = hist_names[i]
            temp_hist_data.sort()
            temp_mean_val = sum(temp_hist_data) / len(temp_hist_data)
            temp_max_val = max(temp_hist_data)
            temp_min_val = min(temp_hist_data)
            temp_middle_val = temp_hist_data[len(temp_hist_data) // 2]
            temp_75_val = temp_hist_data[int(len(temp_hist_data) * 0.75)]
            temp_25_val = temp_hist_data[int(len(temp_hist_data) * 0.25)]
            temp_90_val = temp_hist_data[int(len(temp_hist_data) * 0.9)]
            temp_95_val = temp_hist_data[int(len(temp_hist_data) * 0.95)]
            temp_99_val = temp_hist_data[int(len(temp_hist_data) * 0.99)]
            if is_scale:
                temp_hist_data = [i / scale_factor for i in temp_hist_data]
                temp_mean_val /= scale_factor
                temp_max_val /= scale_factor
                temp_min_val /= scale_factor
                temp_middle_val /= scale_factor
                temp_75_val /= scale_factor
                temp_25_val /= scale_factor
                temp_90_val /= scale_factor
                temp_95_val /= scale_factor
                temp_99_val /= scale_factor
            f.write(temp_name + '\n')
            f.write('Mean: ' + str(temp_mean_val) + '\n')
            f.write('Max: ' + str(temp_max_val) + '\n')
            f.write('Min: ' + str(temp_min_val) + '\n')
            f.write('Middle: ' + str(temp_middle_val) + '\n')
            f.write('25th percentile: ' + str(temp_25_val) + '\n')
            f.write('75th percentile: ' + str(temp_75_val) + '\n')
            f.write('90th percentile: ' + str(temp_90_val) + '\n')
            f.write('95th percentile: ' + str(temp_95_val) + '\n')
            f.write('99th percentile: ' + str(temp_99_val) + '\n')
            f.write('\n')
    f.close()
    print('Log file saved successfully at ' + save_folder + save_file)

def dict_list_to_dataframe(dict_list, key1, key2):
    """
    Converts a list of dictionaries into a pandas DataFrame using two specified keys.
    
    :param dict_list: List of dictionaries.
    :param key1: The first key to use as the first column in the resulting DataFrame.
    :param key2: The second key to use as the second column in the resulting DataFrame.
    :return: A pandas DataFrame with two columns based on the specified keys.
    """
    if not dict_list or not isinstance(dict_list, list) or not all(isinstance(d, dict) for d in dict_list):
        raise ValueError("The input must be a list of dictionaries.")
    
    # Extract the data using the specified keys
    data = [{key1: d.get(key1, None), key2: d.get(key2, None)} for d in dict_list]
    
    # Create a DataFrame from the extracted data
    df = pd.DataFrame(data, columns=[key1, key2])
    
    return df

def get_bar_data(input_list, bin_tuple_list):
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

def create_bin_tuple_list(vals):
    res = []
    for i in range(len(vals)):
        if i == 0:
            res.append((-999, vals[i]))
        elif i == len(vals) - 1:
            res.append((vals[i], 999999))
        else:
            res.append((vals[i], vals[i + 1]))
    return res

def process_data_within_range_for_classes(dicts, start_date, end_date, class_column = 'Classification_V2', if_ignore_background = True):
    """
    Process data based on the given parameters. dicts is a list of dictionaries, each dict use timestamp as the key
    and the value is the data. start_date and end_date are the date range to process. check_key is the key in data 
    (which is the subdict) to check in
    """
    res = {}
    def parse_datetime(dt_str):
        return datetime.strptime(dt_str, '%Y-%m-%d-%H-%M-%S')   
    def parse_date(date_str):
        return datetime.strptime(date_str, '%Y-%m-%d').date()    
    def get_class_unique_num(dicts, class_column):
        # get the unique number of class_column in the list of dicts
        unique_class = list(set([item[class_column] for item in dicts]))
        return len(unique_class)
    start_date = parse_date(start_date)
    end_date = parse_date(end_date)    
    class_num = get_class_unique_num(dicts, class_column) - 1 if if_ignore_background else get_class_unique_num(dicts, class_column)
    
    for item in dicts:
        curr_time = parse_datetime(item['start_time']).date()
        temp_list = [0] * class_num
        if curr_time <= end_date and curr_time >= start_date:
            if not res or curr_time not in res:
                temp_list = [0, 0, 0]
            else:
                temp_list = res[curr_time]
            if item[class_column] == class_num + 1 and if_ignore_background:
                continue
            temp_list[item[class_column] - 1] += 1
            res[curr_time] = temp_list
  
    return res

def find_bin_ind(val, bin_ticks, min_val = 0, max_val = 999999):
    if min_val <= val < bin_ticks[0]:
        return 0
    elif val >= bin_ticks[-1]:
        return len(bin_ticks)
    else:
        for i in range(len(bin_ticks) - 1):
            temp_tick = bin_ticks[i]
            if temp_tick <= val < bin_ticks[i + 1]:
                return i + 1
 
 
def smart_rounding(value):
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
    if magnitude < 0:
        return round(value, -magnitude)
    else:
        power = 10 ** magnitude
        return round(value / power) * power

def generate_bin_ticks(data, num_bins, mode='data', smart_round=False):
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
        bin_edges = np.vectorize(smart_rounding)(bin_edges)
    
    return bin_edges   

def analyze_cross_relationship(data, key1, key2, key1_ticks = None, key2_ticks = None, x_scale = 1, y_scale = 1):
    """
    Analyzes the cross-relationship between two variables in a list of dictionaries.

    :param data: List of dictionaries containing the data.
    :param key1: The first key in the dictionaries to analyze.
    :param key2: The second key in the dictionaries to analyze.
    :return: A dictionary with keys as tuple pairs of values from key1 and key2, and values as their frequency.
    """
    pair_frequency = {}
    
    for entry in data:
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
            key_pair = (find_bin_ind(value1, key1_ticks), find_bin_ind(value2, key2_ticks))
        if key_pair[0] is None:
            print(value1)
        # Increment the count for this key pair in the dictionary
        if key_pair in pair_frequency:
            pair_frequency[key_pair] += 1
        else:
            pair_frequency[key_pair] = 1

    return pair_frequency

def convert_pd_to_dict(df):
    dict_data = {}
    print(df.columns)
    for i in range(len(df)):
        temp_key = df.index[i].strftime('%Y-%m-%d-%H-%M-%S')
        temp_items = {}
        for c in df.columns.to_list():
            temp_item = df[c].iloc[i]
            
            if np.isnan(temp_item):
                break
            temp_items[c] = df[c].iloc[i]
        if not np.isnan(temp_item):
            dict_data[temp_key] = temp_items
    return dict_data

def merge_rip_dict_with_other_data(daily_event_list, other_dict, other_interval, other_keys, 
                                   start_time_str=None, range_limited=True, lag_time=0):
    daily_num = len(daily_event_list)
    res = []
    time_offset = timedelta(hours=lag_time)
    time_format = "%Y-%m-%d-%H-%M-%S"
    
    for i in range(daily_num):
        temp_event = daily_event_list[i]
        
        # Parse and shift timestamps
        adj_start_time = datetime.strptime(temp_event.get('start_time'), time_format) - time_offset
        adj_end_time = datetime.strptime(temp_event.get('end_time'), time_format) - time_offset
        adj_peak_time = datetime.strptime(temp_event.get('peak_time'), time_format) - time_offset

        # Convert back to string if needed by helper functions
        adj_start_str = adj_start_time.strftime(time_format)
        adj_end_str = adj_end_time.strftime(time_format)
        adj_peak_str = adj_peak_time.strftime(time_format)

        # Find closest timestamps
        start_low_timestamp, start_high_timestamp = find_closest_timestamps(adj_start_str, other_interval, start_time_str=start_time_str)
        end_low_timestamp, end_high_timestamp = find_closest_timestamps(adj_end_str, other_interval, start_time_str=start_time_str)
        peak_low_timestamp, peak_high_timestamp = find_closest_timestamps(adj_peak_str, other_interval, start_time_str=start_time_str)

        # Compute weights
        start_weight = compute_weight(adj_start_str, start_low_timestamp, start_high_timestamp)
        end_weight = compute_weight(adj_end_str, end_low_timestamp, end_high_timestamp)
        peak_weight = compute_weight(adj_peak_str, peak_low_timestamp, peak_high_timestamp)

        for k in other_keys:
            if k.lower() in ['direction', 'angle']:
                temp_event['Start_' + k] = weighted_average_angle(
                    other_dict[start_low_timestamp][k],
                    other_dict[start_high_timestamp][k],
                    start_weight,
                    range_limited=range_limited
                )
                temp_event['End_' + k] = weighted_average_angle(
                    other_dict[end_low_timestamp][k],
                    other_dict[end_high_timestamp][k],
                    end_weight,
                    range_limited=range_limited
                )
                temp_event['Peak_' + k] = weighted_average_angle(
                    other_dict[peak_low_timestamp][k],
                    other_dict[peak_high_timestamp][k],
                    peak_weight,
                    range_limited=range_limited
                )
            else:
                temp_event['Start_' + k] = interpolate_vals(
                    other_dict[start_low_timestamp][k],
                    other_dict[start_high_timestamp][k],
                    start_weight
                )
                temp_event['End_' + k] = interpolate_vals(
                    other_dict[end_low_timestamp][k],
                    other_dict[end_high_timestamp][k],
                    end_weight
                )
                temp_event['Peak_' + k] = interpolate_vals(
                    other_dict[peak_low_timestamp][k],
                    other_dict[peak_high_timestamp][k],
                    peak_weight
                )

        res.append(temp_event)
    
    return res