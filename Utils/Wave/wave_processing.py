# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 14:55:09 2023

@author: bbean

"""
import copy
import numpy as np
import datetime
import glob
import os
import re
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import math
from ..DateTime.BasicDateTime import convert_utc_to_central, sort_dates, is_time_ahead, compare_timestamp_order
from .wave_physical_process import wave_routing, new_wave_routing
from .physical_process import wave_routing_by_depth

#%% This is the cell contains functions about interpolation.
# We skipped the case of -999 at this moment since the data does not contain any -999s in our research period.
# If there exist -999s, please interpolate first before processing.

def wave_interpolation(wave_data, start_ind, end_ind, thresh):
    if abs(start_ind - end_ind) > thresh:
        return wave_data
    else:
        for i in range(start_ind, end_ind + 1):
            delta_wave = wave_data[end_ind] - wave_data[start_ind]
            wave_data[i] = wave_data[start_ind] + delta_wave * (i - start_ind) / (end_ind - start_ind)
        return wave_data

def linear_estimator(fore_val, back_val, weight):
    return back_val + (fore_val - back_val) * weight

def nearest_estimator(fore_val, back_val, weight):
    if weight < 0.5:
        return back_val
    elif weight > 0.5:
        return fore_val
    else:
        temp_val = random.random()
        if temp_val >= 0.5:
            return fore_val
        else:
            return back_val

def compute_weight(min_val, sec_val):
    total_sec = min_val * 60 + sec_val
    return total_sec / 3600

def find_nan_vals(wave_data, nan_val = -999):
    return np.where(wave_data == nan_val)[0]

#%% This is the cell contains functions about time-related processing.
def if_leap_year(year):
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)

def get_upper_lower_time(year, month, day, hour, minute, second):
    if minute == 0 and second == 0:
        upper_arr = [year, month, day, hour, minute, second]
        lower_arr = [year, month, day, hour, minute, second]
        weight = 0
    else:
        weight = compute_weight(minute, second)
        if hour < 23:
            upper_arr = [year, month, day, hour + 1, 0, 0]
            lower_arr = [year, month, day, hour, 0, 0]
        else:
            day_arr_1 = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
            day_arr_2 = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
            if_leap = if_leap_year(year)
            day_arr = day_arr_2 if if_leap else day_arr_1
            if hour == 23:
                if day != day_arr[month]:
                    upper_arr = [year, month, day + 1, 0, 0, 0]
                    lower_arr = [year, month, day, hour, 0, 0]
                else:
                    if month != 12:
                        upper_arr = [year, month + 1, 1, 0, 0, 0]
                        lower_arr = [year, month, day, hour, 0, 0]
                    else:
                        upper_arr = [year + 1, 1, 1, 0, 0, 0]
                        lower_arr = [year, month, day, hour, 0, 0]
    return upper_arr, lower_arr, weight

def extract_time_inds(time_data, year, month, day, hr, minute):
    check_time = datetime.datetime(year, month, day, hr, minute)
    is_found = False
    for i in range(len(time_data)):
        if round((time_data[i] - check_time).total_seconds()) == 0:
            res = i
            is_found = True
            break
    if not is_found:
        return -999
    else:
        return res

def extract_data_of_time_interval(time_data, orig_data, start_year, start_month, start_day, end_year, end_month, end_day):
    for i in range(24):
        start_ind = extract_time_inds(time_data, start_year, start_month, start_day, i, 0)
        if start_ind >= 0:
            break
    for j in range(23, -1, -1):
        end_ind = extract_time_inds(time_data, end_year, end_month, end_day, j, 0)
        if end_ind >= 0:
            break
    if start_ind < 0 or end_ind < 0:
        return [], []
    else:
        return orig_data[start_ind: end_ind + 1], time_data[start_ind: end_ind + 1], start_ind, end_ind

def get_all_candidate_time(folder, img_suffix = 'jpg'):
    subfolders = [os.path.join(folder, d) for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]
    res = []
    for s in subfolders:
        search_string = s + '/*.' + img_suffix
        files = glob.glob(search_string)
        for f in files:
            r_pattern = '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]-[0-9][0-9]-[0-9][0-9]-[0-9][0-9]'
            time_str = re.search(r_pattern, f).group()
            time_str_comps = time_str.split('-')
            res.append([str(tsc) for tsc in time_str_comps])
    res = np.array(res)
    return res

def get_wave_data_column(time_arr, time_data, wave_data, choice = 'Linear'):
    # Time_arr is the time read from image data.
    # Time_data is the time read from wave data.
    # Wave_Data is the data to process.
    # Choice is the way to interpolate, can be Linear, or NN.
    # get upper, # get lower, # get weighted
    res = []
    for i in tqdm(range(len(time_arr))):
        #print(i)
        temp_time = time_arr[i, :]
        temp_year, temp_month, temp_day, temp_hr, temp_min, temp_sec = \
            int(temp_time[0]), int(temp_time[1]), int(temp_time[2]), int(temp_time[3]), int(temp_time[4]), int(temp_time[5])
        time_ind =  extract_time_inds(time_data, temp_year, temp_month, temp_day, temp_hr, temp_min)
        if time_ind >= 0:
            res.append(wave_data[time_ind])
        else:
            upper_time, lower_time, weight = get_upper_lower_time(temp_year, temp_month, temp_day, temp_hr, temp_min, temp_sec)
            upper_ind, lower_ind = extract_time_inds(time_data, upper_time[0], upper_time[1], upper_time[2], upper_time[3], upper_time[4]), \
                extract_time_inds(time_data, lower_time[0], lower_time[1], lower_time[2], lower_time[3], lower_time[4])
            fore_val, back_val = wave_data[upper_ind], wave_data[lower_ind]
            if choice == 'Linear':
                temp_val = linear_estimator(fore_val, back_val, weight)
            else:
                temp_val = nearest_estimator(fore_val, back_val, weight)
            res.append(temp_val)
    return res

def convert_np_to_pd(time_arr):
    datetime_strings = [' '.join(row) for row in time_arr]
    datetimes = pd.to_datetime(datetime_strings, format='%Y %m %d %H %M %S')
    df = pd.DataFrame(datetimes, columns=['Datetime'])
    return df
    
def add_new_wave_data_column(df, new_column, new_name):
    df[new_name] = new_column
    return df

def get_binary_columns(wave_table, base_azimuth, ang_thresh, height_thresh):
    res = []
    for i in range(len(wave_table)):
        temp_angle = wave_table.angle[i]
        temp_height = wave_table.hs[i]
        if temp_height < height_thresh:
            res.append('Small_Wave')
        elif (temp_angle < base_azimuth + 90 - ang_thresh) | (temp_angle > base_azimuth + 90 + ang_thresh):
            res.append('Oblique_Wave')
        else:
            res.append('Normal_Wave')
    return res

#%% This cell contains function to process the original dataframe and merge the dataframe based on events, different events will be
#   seperated by time difference.
def merge_image_df(df, threshold):
    # Calculate time differences
    df['TimeDiff'] = df['Datetime'].diff()

    # Identify where splits should occur
    split_indices = df[df['TimeDiff'] > threshold].index

    # Initialize an empty DataFrame for the result
    result_df = pd.DataFrame()

    # Function to compute averages of a DataFrame
    def compute_averages(dataframe):
        return dataframe.mean(numeric_only=True)

    # Split and process the DataFrame
    start_idx = 0
    for end_idx in split_indices:
        # Split the DataFrame
        temp_df = df.loc[start_idx:end_idx-1]

        # Compute and append averages to result DataFrame
        result_df = result_df.append(compute_averages(temp_df), ignore_index=True)
        start_idx = end_idx

    # Process the last segment
    temp_df = df.loc[start_idx:]
    result_df = result_df.append(compute_averages(temp_df), ignore_index=True)

    return result_df

def convert_wave_data_into_dict(wave_timestamp, wave_direction, wave_height, wave_period, azimuth, convert_timezone = True, convert_to_positive = False):
    wave_data = {}
    for i in range(len(wave_timestamp)):
        if convert_timezone:
            time_string = convert_utc_to_central(wave_timestamp[i].strftime('%Y-%m-%d-%H-%M-%S'))
        else:
            time_string = wave_timestamp[i].strftime('%Y-%m-%d-%H-%M-%S')
        temp_wave_direction = wave_direction[i] - (azimuth + 90)
        if convert_to_positive:
            if temp_wave_direction < 0:
                temp_wave_direction += 360
        wave_data[time_string] = {'direction': temp_wave_direction, 'height': wave_height[i], 'period': wave_period[i]}
    return wave_data

def weighted_average_angle(angle1, angle2, weight1, range_limited = True):
    # Convert angles to radians
    angle1_rad = math.radians(angle1)
    angle2_rad = math.radians(angle2)
    
    # Calculate weighted x and y components
    x = weight1 * math.cos(angle1_rad) + (1 - weight1)* math.cos(angle2_rad)
    y = weight1 * math.sin(angle1_rad) + (1 - weight1)* math.sin(angle2_rad)
    
    # Calculate the weighted average angle in radians
    weighted_angle_rad = math.atan2(y, x)
    
    # Convert the weighted average angle back to degrees
    weighted_angle_deg = math.degrees(weighted_angle_rad)
    
    # Ensure the angle is between 0 and 360 degrees
    if weighted_angle_deg < 0:
        weighted_angle_deg += 360
    
    if range_limited:
        if weighted_angle_deg > 180:
            weighted_angle_deg -= 180

    return weighted_angle_deg

#%% This is the cell contains functions about disperse relationship.
def wave_disperse_relation(d, T):
    """

    :param d: depth (m)
    :param T: wave period (s)
    :return: wave length (m), wave speed(m/s)
    """
    g = 9.81

    if d > 0.5 * 1.56 * T**2:
        return 1.56 * T**2, 1.56 * T
    L0 = 0
    L1 = g * T ** 2 / 2 / math.pi

    while np.abs(L0 - L1) > 1e-1:
        L0 = L1
        L1 = g * T**2 / 2 / math.pi * np.tanh(2 * math.pi * d / L0)

    L = L1
    C = L1 / T

    return L, C

def is_deep_water(L, d):
    return d >= 1.5 * L
    
def wave_routing_for_wave_dict(wave_dict):
    sorted_keys = sort_dates(wave_dict)
    deep_water_depth = 44
    res_dict = {}
    for k in sorted_keys:
        temp_wave_comps = wave_dict.get(k)
        deep_water_dir = temp_wave_comps.get('direction')
        deep_water_period = temp_wave_comps.get('period')
        deep_water_height = temp_wave_comps.get('height')

def update_with_wave_routing(final_res, compute_timestamp = 'Start', offshore_depth = 45, kb = 0.78, bottom_slope = 0.04):
    res = []
    for i in tqdm(range(len(final_res))):
        temp_event = final_res[i]
        input_height = temp_event.get('%s_height'%compute_timestamp)
        input_angle = temp_event.get('%s_direction'%compute_timestamp)
        
        input_period = temp_event.get('%s_period'%compute_timestamp)
        if input_angle < 0 or input_angle > 180:
            temp_event['breaking_dist'] = -999
            temp_event['nearshore_angle'] = input_angle
            temp_event['nearshore_height'] = input_height
        else:
            input_angle = 90 - input_angle if input_angle < 90 else input_angle - 90
            [cur_depth, hb, alpha, c, L0] = wave_routing(input_height, input_angle, input_period, s0=bottom_slope, d0 = offshore_depth, kb = kb)
            temp_event['nearshore_angle'] = alpha
            temp_event['nearshore_height'] = hb
            temp_event['breaking_dist'] = cur_depth / bottom_slope
        final_res[i] = temp_event
    return final_res

def update_with_wave_depth_routing(final_res, compute_timestamp = 'Start', offshore_depth = 44, kb = 0.78, steps = 100,
                                   target_depth = 2, bottom_slope = 0.028):
    res = []
    for i in tqdm(range(len(final_res))):
        temp_event = final_res[i]
        input_height = temp_event.get('%s_height'%compute_timestamp)
        input_angle = temp_event.get('%s_direction'%compute_timestamp)
        if (input_angle > 270):
            input_angle = input_angle - 360
        input_period = temp_event.get('%s_period'%compute_timestamp)
        if input_angle <= -90 or input_angle >= 90:
            temp_event['breaking_dist'] = -999
            temp_event['nearshore_angle'] = input_angle
            temp_event['nearshore_height'] = input_height
        else:
            #input_angle = 90 - input_angle if input_angle < 90 else input_angle - 90
            # [cur_depth, hb, alpha, c, L0] = wave_routing(input_height, input_angle, input_period, s0=bottom_slope, d0 = offshore_depth, kb = kb)
            d0 = offshore_depth
            if (kb < 0):
                [output_height, output_time, output_angle] = wave_routing_by_depth(input_height, input_angle, input_period, d0, target_depth=target_depth,steps=steps)
            else:
                [output_height, output_time, output_angle] = wave_routing_by_depth(input_height, input_angle, input_period, d0, target_depth=target_depth, breaking_criteria=kb, steps=steps)
            # wave_routing_by_depth(h0: float,
            #               alpha0: float,
            #               T: float,
            #               d0: float,
            #               target_depth: float = 2,
            #               breaking_criteria: float = 0.8,
            #               steps=100)
            temp_event['nearshore_angle'] = output_angle
            temp_event['nearshore_height'] = output_height
            temp_event['breaking_dist'] = target_depth / bottom_slope
        final_res[i] = temp_event
    return final_res

def update_with_new_wave_routing(final_res, compute_timestamp = 'Start', offshore_depth = 44, check_depth = 2, kb = 0.8):
    res = []
    for i in tqdm(range(len(final_res))):
        temp_event = final_res[i]
        input_height = temp_event.get('%s_height'%compute_timestamp)
        input_angle = temp_event.get('%s_direction'%compute_timestamp)
        
        input_period = temp_event.get('%s_period'%compute_timestamp)
        if input_angle <= -90 or input_angle >= 90:
            temp_event['is_breaking'] = False
            temp_event['nearshore_angle'] = input_angle
            temp_event['nearshore_height'] = input_height
        else:
            [h, T, alpha] = wave_routing_by_depth(input_height, input_angle, input_period, d0 = offshore_depth, breaking_criteria = kb, target_depth = check_depth)
            #temp_event['is_breaking'] = is_break
            temp_event['nearshore_angle'] = alpha
            temp_event['nearshore_height'] = h
        final_res[i] = temp_event
    return final_res

def add_routing_to_wave_data(wave_dict,  offshore_depth = 44, kb = 0.8, bottom_slope = 0.028, dist_thresh = 20, target_depth=0.5, steps = 10000):
    res = {}
    for k in tqdm(wave_dict.keys()):
        temp_content = wave_dict.get(k)
        alpha0, height, period = temp_content.get('direction'), temp_content.get('height'), temp_content.get('period')
        if alpha0 <= -90 or alpha0 >= 90:
            # temp_content['breaking_dist'] = np.nan
            temp_content['nearshore_angle'] = np.nan
            temp_content['nearshore_height'] = np.nan
            temp_content['breaking_depth'] = np.nan
            # temp_content['wave_length'] = np.nan
            # temp_content['wave_speed'] = np.nan
            # temp_content['breaking_dist'] = -999
            # temp_content['is_breaking'] = False
        else:
            [h, T, alpha] = wave_routing_by_depth(height, abs(alpha0), period, d0 = offshore_depth, breaking_criteria = kb, target_depth = target_depth, steps=steps)
            #temp_event['is_breaking'] = is_break
            
            # cur_dist = cur_depth / bottom_slope
            # temp_content['breaking_dist'] = cur_dist
            temp_content['nearshore_angle'] = alpha
            temp_content['nearshore_height'] = h
            # temp_content['breaking_depth'] = cur_depth
            # temp_content['wave_length'] = L
            # temp_content['wave_speed'] = c
            # temp_content['breaking_dist'] = cur_depth / bottom_slope
            # temp_content['is_breaking'] = cur_dist > dist_thresh
        res[k] = temp_content
    return res

def add_missing_to_wave_data(wave_dict, missing_dict, start_date = '2019-05-15', end_date = '2019-09-14', start_time = '08-00-00', end_time = '18-00-00'):
    res = {}
    start_datetime = start_date + '-' + start_time
    end_datetime = end_date + '-' + end_time

    for k in tqdm(wave_dict.keys()):
        temp_datetime = str(k)
        if not is_time_ahead(temp_datetime, start_datetime) or is_time_ahead(temp_datetime, end_datetime):
            continue
        else:
            temp_time = '-'.join(temp_datetime.split('-')[3:])
            temp_date = '-'.join(temp_datetime.split('-')[:3])
            if compare_timestamp_order(temp_time, start_time, include = False):
                continue
            elif compare_timestamp_order(end_time, temp_time, include = True):
                continue
            else:
                temp_content = wave_dict.get(k)
                temp_missing_dict = missing_dict.get(temp_date)
                if not temp_missing_dict or k not in temp_missing_dict:
                    temp_content['is_missing'] = True
                else:
                    temp_content['is_missing'] = False
                res[k] = temp_content
    return res
