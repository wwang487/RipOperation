from datetime import datetime, timedelta,time
import pandas as pd
import pytz
import os
from tqdm import tqdm
import numpy as np

def sort_dates(dictionary, format = "%Y-%m-%d-%H-%M-%S"):
    # Convert string keys to datetime objects
    date_keys = [datetime.strptime(key, format) for key in dictionary.keys()]
    # Sort the datetime objects
    sorted_dates = sorted(date_keys)
    # Convert sorted datetime objects back to strings
    sorted_date_strings = [date.strftime(format) for date in sorted_dates]
    return sorted_date_strings

def time_difference(datetime_str1, datetime_str2, format1 = "%Y-%m-%d-%H-%M-%S", format2 = "%Y-%m-%d-%H-%M-%S"):
    # Convert the strings to datetime objects
    datetime_obj1 = datetime.strptime(datetime_str1, format1)
    datetime_obj2 = datetime.strptime(datetime_str2, format2)

    # Calculate the time difference
    difference = datetime_obj1 - datetime_obj2

    # You can format the difference as needed, here it's returned as days, seconds, and microseconds
    return difference

def convert_np_to_pd(time_arr):
    datetime_strings = [' '.join(row) for row in time_arr]
    datetimes = pd.to_datetime(datetime_strings, format='%Y %m %d %H %M %S')
    df = pd.DataFrame(datetimes, columns=['Datetime'])
    return df

def get_first_last_dates(series, format = '%04d-%02d-%02d'):
    """
    Returns the first and last date of a pandas Series with datetime index.
    
    Parameters:
    - series: pandas Series object with a datetime index.
    
    Returns:
    - Tuple containing the first and last date of the series index.
    """
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("The series index must be a DatetimeIndex.")
        
    first_date = series.index.min()
    last_date = series.index.max()
    
    return format%(first_date.year, first_date.month, first_date.day), format%(last_date.year, last_date.month, last_date.day)

def add_seconds_to_timestamp(timestamp_str, seconds, timestamp_format = '%Y-%m-%d-%H-%M-%S'):
    # Convert the timestamp string to a datetime object
    timestamp = datetime.strptime(timestamp_str, timestamp_format)
    
    # Add the specified number of seconds to the datetime object
    new_timestamp = timestamp + timedelta(seconds=seconds)
    
    # Convert the new datetime object back to a string in the specified format
    new_timestamp_str = new_timestamp.strftime(timestamp_format)
    
    return new_timestamp_str

def is_time_ahead(timestamp_str1, timestamp_str2, timestamp_format = '%Y-%m-%d-%H-%M-%S'):
    # Convert the timestamp strings to datetime objects
    timestamp1 = datetime.strptime(timestamp_str1, timestamp_format)
    timestamp2 = datetime.strptime(timestamp_str2, timestamp_format)
    
    # Compare the two datetime objects
    return timestamp1 > timestamp2

def convert_utc_to_central(timestamp_str, timestamp_format = '%Y-%m-%d-%H-%M-%S'):
    
    # Create a datetime object from the input timestamp string
    utc_time = datetime.strptime(timestamp_str, timestamp_format)
    
    # Define the UTC and Central Time zones
    utc_zone = pytz.utc
    central_zone = pytz.timezone('US/Central')
    
    # Localize the datetime object to UTC
    utc_time = utc_zone.localize(utc_time)
    
    # Convert the datetime object from UTC to Central Time
    central_time = utc_time.astimezone(central_zone)
    
    # Return the datetime object in the desired format
    return central_time.strftime(timestamp_format)

def convert_and_sort_datetime_index(datetime_index, convert_timezone = True, timestamp_format = '%Y-%m-%d-%H-%M-%S'):
    # Convert the datetime index to a list of formatted strings
    if convert_timezone:
        formatted_timestamps = [convert_utc_to_central(dt.strftime(timestamp_format)) for dt in datetime_index]
    else:
        formatted_timestamps = [dt.strftime(timestamp_format) for dt in datetime_index]
    
    # Sort the list of formatted timestamps
    formatted_timestamps.sort()
    
    return formatted_timestamps

def convert_and_sort_date_and_time_index(dates, times, convert_timezone = True):
    # Define the format for the output timestamp strings
    timestamp_format = '%Y-%m-%d-%H-%M-%S'
    
    input_len = len(dates)

    # Convert the datetime index to a list of formatted strings
    if convert_timezone:
        formatted_timestamps = [convert_utc_to_central(datetime.strptime(dates[i] + times[i])) for i in range(input_len)]
    else:
        formatted_timestamps = [datetime.strptime(dates[i] + times[i], timestamp_format) for i in range(input_len)]
    
    # Sort the list of formatted timestamps
    formatted_timestamps.sort()
    
    return formatted_timestamps

def check_missing_timestamps(input_data, start_time, end_time, interval):
    curr_time = start_time
    res = []
    while not is_time_ahead(curr_time, end_time):
        
        if curr_time not in input_data:
            res.append(curr_time)
        curr_time = add_seconds_to_timestamp(curr_time, interval)
    return res

def convert_data_to_pd_dict(input_dict, key_name):
    res = {}
    for k, v in input_dict.items():
        key_str = str(k)
        y, m, d, h, mi = (int(i) for i in key_str.split('-')[:-1])
        res[datetime(y, m, d, h, mi)] = v[key_name]
    return res

def convert_data_to_pd_dict_within_range(input_dict, key_name, start_time, end_time, is_feet = False):
    start_dt = datetime.strptime(start_time, '%Y-%m-%d-%H-%M-%S')
    end_dt = datetime.strptime(end_time, '%Y-%m-%d-%H-%M-%S')
    
    res = {}
    for k, v in input_dict.items():
        key_dt = datetime.strptime(k, '%Y-%m-%d-%H-%M-%S')
        if start_dt <= key_dt <= end_dt:
            key_str = str(k)
            y, m, d, h, mi = (int(i) for i in key_str.split('-')[:-1])
            if not is_feet:
                res[datetime(y, m, d, h, mi)] = v.get(key_name)
            else:
                res[datetime(y, m, d, h, mi)] = v.get(key_name) * 0.3048
    return res

def adjust_timestamp(timestamp_str, interval_seconds, check_before):
    # Define the format of the input and output timestamp string
    timestamp_format = '%Y-%m-%d-%H-%M-%S'
    
    # Convert the timestamp string to a datetime object
    timestamp = datetime.strptime(timestamp_str, timestamp_format)
    
    # Adjust the timestamp based on the dummy variable
    if check_before:
        new_timestamp = timestamp - timedelta(seconds=interval_seconds)
    else:
        new_timestamp = timestamp + timedelta(seconds=interval_seconds)
    
    # Return the new timestamp as a string in the same format
    return new_timestamp.strftime(timestamp_format)

def find_closest_timestamps(timestamp_str, interval_seconds, start_time_str=None, timestamp_format='%Y-%m-%d-%H-%M-%S'):
    # Define the format of the input timestamp string
    # timestamp_format = '%Y-%m-%d-%H-%M-%S'
    
    # Convert the timestamp string to a datetime object
    timestamp = datetime.strptime(timestamp_str, timestamp_format)
    
    # Set default start time to midnight of the given day if not specified
    if start_time_str is None:
        start_time = timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
    else:
        # Parse the custom start time string
        start_time = datetime.strptime(start_time_str, timestamp_format)
    
    # Calculate the number of seconds from the custom start time or midnight
    seconds_since_start = (timestamp - start_time).total_seconds()
    
    # Find the closest lower multiple of the interval
    lower_multiplier = int(seconds_since_start // interval_seconds)
    lower_timestamp = start_time + timedelta(seconds=lower_multiplier * interval_seconds)
    
    # Find the closest higher multiple of the interval
    upper_multiplier = lower_multiplier + 1
    upper_timestamp = start_time + timedelta(seconds=upper_multiplier * interval_seconds)
    
    # Convert the lower and upper timestamps back to strings
    lower_timestamp_str = lower_timestamp.strftime(timestamp_format)
    upper_timestamp_str = upper_timestamp.strftime(timestamp_format)
    
    return lower_timestamp_str, upper_timestamp_str

def parse_time(time_str):
    """Parse a string formatted as hh-mm-ss to a datetime.time object."""
    h, m, s = map(int, time_str.split('-'))
    return datetime.strptime(f"{h}:{m}:{s}", "%H:%M:%S").time()

def overlaps(event_start, event_end, interval_start, interval_end):
    """Check if the event overlaps with a given interval."""
    return max(event_start, interval_start) < min(event_end, interval_end)

def get_time_range_histogram_data(entries, start_counting_time='00-00-00', end_counting_time='23-59-59', interval=30):
    start_time = parse_time(start_counting_time)
    end_time = parse_time(end_counting_time)
    interval_delta = timedelta(minutes=interval)
    histogram = {}

    # Create interval ranges for a day
    intervals = []
    base_date = datetime.today().date()  # Use any arbitrary date
    current_time = datetime.combine(base_date, start_time)
    while current_time.time() < end_time:
        interval_start = current_time
        current_time += interval_delta
        intervals.append((interval_start, min(current_time, datetime.combine(base_date, end_time))))

    # Count overlaps for each interval
    for interval_start, interval_end in intervals:
        count = 0
        for entry in entries:
            event_start = datetime.strptime(entry['start_time'], '%Y-%m-%d-%H-%M-%S')
            event_end = datetime.strptime(entry['end_time'], '%Y-%m-%d-%H-%M-%S')

            # Ensure we compare datetime with datetime
            interval_start_time = datetime.combine(event_start.date(), interval_start.time())
            interval_end_time = datetime.combine(event_start.date(), interval_end.time())

            if overlaps(event_start, event_end, interval_start_time, interval_end_time):
                count += 1
        
        histogram[f"{interval_start.time().strftime('%H:%M:%S')} - {interval_end.time().strftime('%H:%M:%S')}"] = count

    return histogram



def format_time(time_str):
    """Convert a string in HH-MM format to a datetime.time object."""
    return datetime.strptime(time_str, "%H-%M").time()

def find_missing_timestamps(base_dir, start_time='08-00', end_time='18-00', interval=15):
    start_time = format_time(start_time)
    end_time = format_time(end_time)
    expected_images_per_hour = 3600 // interval
    missing_timestamps = {}  # Dictionary to store missing timestamps

    # List directories to estimate total progress accurately
    directories = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

    for folder_name in tqdm(directories, desc="Processing folders"):
        folder_path = os.path.join(base_dir, folder_name)
        all_images = [f for f in os.listdir(folder_path) if f.endswith('-PW.jpg')]
        image_times = set(datetime.strptime(img[:-7], '%Y-%m-%d-%H-%M-%S') for img in all_images)

        # Start and end datetime for the day
        day_date = datetime.strptime(folder_name, '%Y-%m-%d')
        start_datetime = datetime.combine(day_date, start_time)
        end_datetime = datetime.combine(day_date, end_time)

        # Prepare timestamps to progress over
        all_timestamps = []
        current_time = start_datetime
        while current_time < end_datetime:
            all_timestamps.append(current_time)
            current_time += timedelta(hours=1)

        missing_day = []  # List to collect missing timestamps for the day

        for current_time in tqdm(all_timestamps, desc=f"Checking {folder_name}", leave=False):
            hour_end = min(current_time + timedelta(hours=1), end_datetime)
            # Count images within this hour
            image_count = sum(1 for img_time in image_times if current_time <= img_time < hour_end)
            if image_count < expected_images_per_hour:
                # If fewer images than expected, check each timestamp
                while current_time < hour_end:
                    next_time = current_time + timedelta(seconds=interval)
                    if not any(current_time <= img_time < next_time for img_time in image_times):
                        missing_day.append(current_time.strftime('%Y-%m-%d-%H-%M-%S'))
                    current_time = next_time
            else:
                # Skip detailed checking if the hour is fully covered
                current_time = hour_end

        missing_timestamps[folder_name] = missing_day

    return missing_timestamps

def compare_timestamp_order(time1, time2, include = True):
    # Parse the time strings into datetime.time objects
    t1 = datetime.strptime(time1, "%H-%M-%S").time()
    t2 = datetime.strptime(time2, "%H-%M-%S").time()
    
    # Compare and print which is earlier or if they are the same
    if include:
        if t1 <= t2:
            return True
        else:
            return False
    else:
        if t1 < t2:
            return True
        else:
            return False
        
def parse_datetime(date_str, time_str):
    """Combines a date and time string into a single datetime object."""
    return datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H-%M-%S")

def find_missing_ranges(missing_timestamps, start_time='08-00-00', end_time='18-00-00', interval=15):
    """
    Finds and groups missing timestamp ranges for each day.
    
    Args:
    missing_timestamps (dict): Dictionary with days as keys and lists of missing timestamps as values.
    start_time (str): The start time of the active period each day.
    end_time (str): The end time of the active period each day.
    interval (int): The interval in seconds to determine breaks between missing timestamp groups.
    
    Returns:
    dict: Dictionary with days as keys and lists of tuple pairs indicating missing ranges.
    """
    missing_ranges = {}
    
    

    for day, times in missing_timestamps.items():
        full_day_start = parse_datetime(day, "00-00-00")
        full_day_end = parse_datetime(day, "23-59-59")
        daily_start_count_time, daily_end_count_time = parse_datetime(day, start_time), parse_datetime(day, end_time)
        # Convert recorded times to datetime objects and sort them
        recorded_times = sorted(parse_datetime(day, time) for time in times)
        
        # Initialize list of ranges
        ranges = []
        if recorded_times:
            start_range = recorded_times[0]
            end_range = start_range

            # Group timestamps into ranges
            for time in recorded_times[1:]:
                if (time - end_range).total_seconds() > interval:
                    ranges.append((start_range, end_range))
                    start_range = time
                end_range = time
            ranges.append((start_range, end_range))  # Append the last range

        # Include entire period before and after if there are no timestamps
        if not recorded_times:
            
            ranges.append((full_day_start, daily_start_count_time - timedelta(seconds=1)))
            ranges.append((daily_end_count_time, full_day_end))
        elif ranges:
            # Append missing ranges before the first and after the last recorded timestamp
            day_start = parse_datetime(day, start_time)
            day_end = parse_datetime(day, end_time)
            if day_start < ranges[0][0]:
                ranges.insert(0, (full_day_start, day_start - timedelta(seconds=1)))
            elif day_start == ranges[0][0]:
                ranges[0] = (full_day_start, ranges[0][1])
            if day_end > ranges[-1][1]:
                ranges.append((day_end, full_day_end))
            elif day_end == ranges[-1][1]:
                ranges[-1] = (ranges[-1][0], full_day_end)
        missing_ranges[day] = ranges

    return missing_ranges

def convert_missing_datetime_to_time(missing_dict):
    res = {}
    for k in missing_dict.keys():
        temp_content = missing_dict.get(k)
        if temp_content:
            temp_res = ['-'.join(i.split('-')[3:]) for i in temp_content]
        else:
            temp_res = []
        res[k] = temp_res
    return res

def convert_missing_match_to_list(missing_match_range):
    res = []
    for k in missing_match_range.keys():
        temp_content = missing_match_range.get(k)
        if temp_content:
            res = res + temp_content
    return res

def merge_close_ranges(missing_ranges, merge_threshold=timedelta(seconds=2)):
    """
    Merges close or overlapping missing ranges.

    Args:
    missing_ranges (list of tuples): List of tuples, each containing start and end datetimes.
    merge_threshold (timedelta): Maximum gap between ranges to consider for merging.

    Returns:
    list of tuples: Merged list of missing ranges.
    """
    if not missing_ranges:
        return []

    # Sort ranges by the start time
    sorted_ranges = sorted(missing_ranges, key=lambda x: x[0])
    
    merged_ranges = [sorted_ranges[0]]
    
    for current_start, current_end in sorted_ranges[1:]:
        last_start, last_end = merged_ranges[-1]
        
        # Check if the current range overlaps or is very close to the last range
        if current_start <= last_end + merge_threshold:
            # Extend the last range
            merged_ranges[-1] = (last_start, max(last_end, current_end))
        else:
            # Add the current range as it is far enough from the last range
            merged_ranges.append((current_start, current_end))
    
    return merged_ranges

def generate_all_timestamps(start, end, interval_seconds):
    """Generates all timestamps from start to end at specified second intervals."""
    current = start
    while current <= end:
        yield current
        current += timedelta(seconds=interval_seconds)


def find_non_missing_timestamps(total_start, total_end, missing_ranges, interval_seconds=60):
    """
    Finds non-missing timestamp ranges that are not within the specified missing ranges.
    
    Args:
    total_start (datetime): The start datetime of the total range.
    total_end (datetime): The end datetime of the total range.
    missing_ranges (list of tuples): Each tuple contains the start and end datetimes of a missing range.
    interval_seconds (int): The interval in seconds for generating timestamps within the total range.
    
    Returns:
    list: A list of tuples, each containing the start and end datetimes of non-missing ranges.
    """
    res = []
    if total_start < missing_ranges[0][0]:
        res.append((total_start, missing_ranges[0][0]))
    for i in range(len(missing_ranges) - 1):
        res.append((missing_ranges[i][1], missing_ranges[i+1][0]))
    if total_end > missing_ranges[-1][1]:
        res.append((missing_ranges[-1][1], total_end))
    return res


def modify_values_outside_hours(data_dict):
    new_dict = {}
    for key, value in data_dict.items():
        # Copy the value or set to np.nan based on the time condition
        if time(8, 0) <= key.time() <= time(18, 0):
            new_dict[key] = value  # keep original value during the hours of 8:00 AM to 6:00 PM
        else:
            new_dict[key] = np.nan  # set to np.nan outside these hours
    return new_dict

def get_daily_ratio_values(input_series, value_1 = -45, value_2 = 45):
    input_series = pd.Series(input_series)
    in_range = input_series.between(value_1, value_2)

    # Group by day and calculate the ratio of values in range vs total valid values (non-NaN)
    ratio_per_day = in_range.groupby(input_series.index.date).mean() *100

    # Convert the index back to datetime if needed
    ratio_per_day.index = pd.to_datetime(ratio_per_day.index)

    # Display the result
    return ratio_per_day.to_dict()