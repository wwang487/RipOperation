from ..DateTime.BasicDateTime import time_difference
import datetime
import pandas as pd

def compute_center_points(date_dictionary):
    res = {}
    for k in date_dictionary.keys():
        temp_xcenter, temp_ycenter = [], []
        for i in range(len(date_dictionary[k])):
            temp_xcenter.append((date_dictionary[k][i]['xmin'] + date_dictionary[k][i]['xmax'])/2)
            temp_ycenter.append((date_dictionary[k][i]['ymin'] + date_dictionary[k][i]['ymax'])/2)
        res[k] = {'x_center': temp_xcenter, 'y_center': temp_ycenter}
    return res

def compute_max_dist(date_dictionary):
    res = {}
    for k in date_dictionary.keys():
        temp_max_dist = 0
        for i in range(len(date_dictionary[k])):
            temp_dist = date_dictionary[k][i]['ymax']
            if temp_dist > temp_max_dist:
                temp_max_dist = temp_dist
        res[k] = temp_max_dist
    return res

def split_events_based_on_max_dist(max_dist_df, sorted_date, time_thresh = 600, dist_percent_thresh = 0.5):
    # if the time difference is larger than a threshold, if the rip current max distance changes a lot, then it's a new event.
    process_num = len(sorted_date)
    final_events = []
    temp_event = {}
    temp_max_dist = -1
    for p_n in range(process_num):
        temp_date = sorted_date[p_n]
        temp_dist = max_dist_df.get(temp_date)
        if temp_max_dist == -1:
            if len(final_events) == 0:
                temp_max_dist = temp_dist
                temp_event['max_dist'] = temp_max_dist
                temp_event['end_dist'] = temp_max_dist
                temp_event['end_time'] = temp_date
                temp_event['start_time'] = temp_date
                temp_event['peak_time'] = temp_date
            elif temp_dist < final_events[-1].get('end_dist') and abs(time_difference(temp_date, final_events[-1].get('end_time')).seconds) < time_thresh:
                continue
            else:
                temp_max_dist = temp_dist
                temp_event['max_dist'] = temp_max_dist
                temp_event['end_dist'] = temp_max_dist
                temp_event['end_time'] = temp_date
                temp_event['start_time'] = temp_date
                temp_event['peak_time'] = temp_date
        else:
            if abs(time_difference(temp_date, temp_event.get('end_time')).seconds) > time_thresh:
                final_events.append(temp_event)
                temp_max_dist = -1
                temp_event = {}
            else:
                if temp_dist > temp_event.get('max_dist'):
                    temp_event['peak_time'] = temp_date
                    temp_event['max_dist'] = temp_dist
                    temp_event['end_dist'] = temp_dist
                    temp_event['end_time'] = temp_date
                elif temp_dist < temp_event.get('max_dist') * dist_percent_thresh:
                    final_events.append(temp_event)
                    temp_max_dist = -1
                    temp_event = {}
                else:
                    temp_event['end_dist'] = temp_dist
                    temp_event['end_time'] = temp_date
    return final_events 

def convert_box_dict_to_series(data_dict, original_format):
    
    # Process the dictionary keys to convert them into a standard datetime format
    new_keys = [pd.to_datetime(key, format = original_format) for key in data_dict.keys()]
    
    # Create the Pandas Series
    series = pd.Series(data=list(data_dict.values()), index=new_keys)
    
    return series
