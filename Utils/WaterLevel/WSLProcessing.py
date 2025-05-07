import pandas as pd
import os
from datetime import datetime
import scipy.signal as signal
import numpy as np
import copy
from ..DateTime.BasicDateTime import convert_utc_to_central, adjust_timestamp, find_closest_timestamps, is_time_ahead
import pytz

#%% This function reads the water level data from a file and returns a DataFrame with specific columns.
def read_and_process_file(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Assuming the first column is 'Date', the second is 'Time', and the last is 'Water Level'
    df['Datetime'] = pd.to_datetime(df.iloc[:, 0] + ' ' + df.iloc[:, 1])
    df['WaterLevel'] = pd.to_numeric(df.iloc[:, -1], errors='coerce')

    # Keep only the 'Datetime' and 'Water Level' columns
    return df[['Datetime', 'WaterLevel']]

# This function returns the water level files in a folder that are located within a certain time range
def filter_files(folder_path, start_year, end_year, station_id, dataset_name):
    # Filter files based on the specified criteria
    filtered_files = []
    for file in os.listdir(folder_path):
        if file.endswith('.csv'):
            parts = file.split('-')
            year, month, dataset, station = int(parts[0]), int(parts[1]), parts[2], parts[3]

            if (start_year <= year <= end_year) and (station_id in file) and (dataset_name in file):
                filtered_files.append(file)
    
    return filtered_files

# This function processes the files in a folder that are within a certain time range, and returns a DataFrame
def process_folder(folder_path, start_time, end_time, station_id, dataset_name):
    start_year = pd.to_datetime(start_time).year
    end_year = pd.to_datetime(end_time).year

    # Get the list of files that match the criteria
    all_files = filter_files(folder_path, start_year, end_year, station_id, dataset_name)

    # Process each file and concatenate the results
    all_data = pd.concat([read_and_process_file(os.path.join(folder_path, f)) for f in all_files])

    # Filter the data based on the specified time range
    filtered_data = all_data[(all_data['Datetime'] >= start_time) & (all_data['Datetime'] <= end_time)]

    return filtered_data


#%% This function is a high pass filter to process the raw water level data
def high_pass_filter(df, cutoff_frequency, order, sampling_rate):
    # Design the Butterworth high-pass filter
    b, a = signal.butter(order, cutoff_frequency, btype='high', fs=sampling_rate)

    # Apply the filter
    df['FilteredWaterLevel'] = signal.filtfilt(b, a, df['WaterLevel'])

    return df

# This function works for merging the Date and TimeStamp columns in the WSL data, and renamed the new column as Datetime
def PreprocessWSLDF(WSL_DF, is_convert_time_zone=False):
    df = copy.deepcopy(WSL_DF)

    # Merge 'Date' and 'TimeStamp' into a single 'Datetime' column
    df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['TimeStamp'])

    # Optionally convert time zone
    if is_convert_time_zone:
        # Assume the original datetime is in UTC
        utc_zone = pytz.timezone('UTC')
        central_zone = pytz.timezone('America/Chicago')

        # Convert to US/Chicago Central Time
        df['Datetime'] = df['Datetime'].dt.tz_localize(utc_zone).dt.tz_convert(central_zone).dt.tz_localize(None)

    # Drop the original 'Date' and 'TimeStamp' columns
    df = df.drop(['Date', 'TimeStamp'], axis=1)

    # Reorder the DataFrame to place 'Datetime' first
    column_order = ['Datetime'] + [col for col in df.columns if col != 'Datetime']
    df = df[column_order]
    
    return df

def convert_WSL_pd_to_dict(wsl_pd, convert_timezone = True):
    wsl_dict = {}
    for i in range(len(wsl_pd)):
        temp_row = wsl_pd.iloc[i]
        time_string = '-'.join(temp_row['Date'].split('/')) + '-' + '-'.join(temp_row['TimeStamp'].split(':')) + '-00'
        if convert_timezone:
            time_string = convert_utc_to_central(time_string)
        wsl_dict[time_string] ={'WSL': temp_row['WSL'], 'filtered_WSL': temp_row['filtered_WSL']}
        
    return wsl_dict

def compute_WLF(rip_event, WSL_dict, WSL_interval, is_feet = True):
    check_time = rip_event.get('start_time')
    start_timestamp = adjust_timestamp(check_time, WSL_interval, True)
    WSL_time_upper1, WSL_time_upper2 = find_closest_timestamps(start_timestamp, WSL_interval)
    WSL_time_lower1, WSL_time_lower2 = find_closest_timestamps(check_time, WSL_interval)
    WSL_start_time, WSL_end_time = WSL_time_upper1, WSL_time_lower2
    curr_time = WSL_start_time
    WSL_record = []
    while not is_time_ahead(curr_time, WSL_end_time):
        WSL_record.append(WSL_dict[curr_time]['WSL'])
        curr_time = adjust_timestamp(curr_time, WSL_interval, False)
    if is_feet:
        return (max(WSL_record) - min(WSL_record)) * 0.3048
    else:
        return max(WSL_record) - min(WSL_record)

def add_WLF_to_rip_dict(rip_event_list, WSL_dict, WSL_interval):
    res = []
    for r in rip_event_list:
        r['WLF'] = compute_WLF(r, WSL_dict, WSL_interval)
        res.append(r)
    return res