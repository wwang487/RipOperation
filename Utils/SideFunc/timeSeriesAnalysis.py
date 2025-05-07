from scipy.signal import butter, filtfilt
import numpy as np
import pandas as pd
import tqdm

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs  # Nyquist Frequency
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def process_zero_crossing(series, start_date, end_date, interval_b, choice = 2):
    # if choice == 1, we calculate zero-crossings one by one
    # if choice == 2, we calculate the highest difference between peak and valley
    # Convert start and end dates to datetime
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # Extend the range for filtering
    extended_start = start_date - pd.Timedelta(minutes=interval_b)
    extended_end = end_date + pd.Timedelta(minutes=interval_b)

    # Filter the original Series based on the extended start and end dates
    extended_series = series[extended_start:extended_end]

    # Initialize the new time series
    new_series = pd.Series(index=pd.date_range(start=start_date, end=end_date, freq=series.index.freq))

    # Process each interval B
    for i in range(len(extended_series) - interval_b):
        current_time = extended_series.index[i]
        subset = extended_series.iloc[i:i + interval_b]
        values = subset.values
        # print(type(values))
        # Find zero crossings
        
        crossings = np.where(np.diff(np.signbit(values)))[0]
        
        # Calculate differences and select the highest
        if choice == 1:
            highest_diff = 0
            for j in range(len(crossings) - 1):
                peak = max(values[crossings[j]:crossings[j + 1] + 1])
                valley = min(values[crossings[j]:crossings[j + 1] + 1])
                diff = abs(peak - valley)
                highest_diff = max(highest_diff, diff)
        else:
            highest_diff = max(values) - min(values)

        # Only update new_series if current_time is within the original date range
        if start_date <= current_time <= end_date:
            new_series.at[current_time] = highest_diff

    return new_series

def insert_missing_timestamps(df, frequency='1T'):
    """
    Function to insert missing timestamps and handle 'M' values correctly.
    """
    df.index = pd.to_datetime(df.index)
    
    # Create a complete range of timestamps based on the specified frequency
    full_time_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq=frequency)
    
    # Reindex the dataframe to the full time range, introducing NaNs for missing rows
    new_df = df.reindex(full_time_range)
    
    # Iterate over the missing timestamps in the new_df
    for time in tqdm(new_df.index[new_df.isna().any(axis=1)]):
        # Find the closest timestamps before and after the current missing timestamp
        before = df.index[df.index < time].max() if any(df.index < time) else None
        after = df.index[df.index > time].min() if any(df.index > time) else None
        
        # If both a 'before' and 'after' time are found, interpolate
        if pd.notna(before) and pd.notna(after):
            weight = (time - before) / (after - before)
            interpolated_values = df.loc[before] + (df.loc[after] - df.loc[before]) * weight
            
            # Handle 'M' values: if either adjacent value was 'M', set the new value to 'M'
            for column in interpolated_values.index:
                if df.loc[before, column] == 'M' or df.loc[after, column] == 'M':
                    new_df.loc[time, column] = 'M'
                else:
                    new_df.loc[time, column] = interpolated_values[column]
                    
    return new_df