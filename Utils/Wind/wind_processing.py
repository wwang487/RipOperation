import os
import pandas as pd
import datetime
from ..DateTime.BasicDateTime import convert_utc_to_central
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
import copy


def read_wind_data_V1(folder, filename, skip_first_row=True):
    """
    Reads a weather data file and returns a DataFrame with specific columns.

    :param folder: The folder where the file is located.
    :param filename: The name of the file to read.
    :return: A DataFrame with columns: Datetime, WDIR, WSPD, and GST.
    """
    # Construct the full file path
    file_path = os.path.join(folder, filename)

    # Read the file into a DataFrame
    df = pd.read_csv(file_path, delim_whitespace=True, usecols=['YY', 'MM', 'DD', 'hh', 'mm', 'WDIR', 'WSPD', 'GST', 'PRES'])
    df = df.iloc[1:, :]
    date_arr = []
    # Combine the date and time columns into a single datetime column
    datetime_dict = {'year':df['YY'], 'month':df['MM'], 'day':df['DD'], 'hour':df['hh'], 'minute':df['mm']}
    
    for i in range(1, len(df) + 1):
        date_arr.append(datetime.datetime(int(datetime_dict['year'][i]), int(datetime_dict['month'][i]), \
            int(datetime_dict['day'][i]), int(datetime_dict['hour'][i]), int(datetime_dict['minute'][i])))
    df['Datetime'] = pd.to_datetime(date_arr)

    # Select only the required columns
    df = df[['Datetime', 'WDIR', 'WSPD', 'GST', 'PRES']]

    return df

def convert_wind_data_into_dict(wind_df, convert_timezone = True):
    wind_dict = {}
    column_names = wind_df.columns
    for i in range(len(wind_df)):
        temp_row = wind_df.iloc[i]
        temp_timestamp = temp_row['Datetime']
        # initial format is like this: 2019-01-01T01:00:00.000, convert to yyyy-mm-dd-hh-MM-ss
        time_string = temp_timestamp.strftime('%Y-%m-%d-%H-%M-%S')
        if convert_timezone:
            time_string = convert_utc_to_central(time_string)
        temp_dict = {}
        for column_name in column_names:
            temp_dict[column_name] = float(temp_row[column_name]) if column_name != 'Datetime' else temp_row[column_name]
        wind_dict[time_string] = temp_dict        
    return wind_dict



def read_wind_data_V2(wind_folder, wind_file, skip_first_row = True, is_interpolate = True, 
                      use_cols = ['valid', 'sknt', 'gust', 'drct', 'alti'],
                      is_convert_time_zone = True, cols_to_interpolate = ['sknt', 'alti', 'drct'], 
                      start_time = None, end_time = None, is_adjust_time = True,
                      is_convert_from_knot = True, is_convert_from_iiHg = True):
    
    file_path = os.path.join(wind_folder, wind_file)

    if skip_first_row:
        df = pd.read_csv(file_path, skiprows=0, usecols=use_cols)
    else:
        df = pd.read_csv(file_path, usecols=use_cols)
    df.rename(columns={'valid': 'DateTime'}, inplace=True)
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    # Assuming 'df' is your DataFrame
    
    for column in df.columns:
        if column != 'DateTime':  # Skip the 'DateTime' column
            df[column] = pd.to_numeric(df[column], errors='coerce')
    
    # Ensure 'DateTime' is the index
    if df.index.name != 'DateTime':
        df.set_index('DateTime', inplace=True)
    
    if is_interpolate:
        df[cols_to_interpolate] = df[cols_to_interpolate].interpolate(method='linear')
    
    if start_time is not None and end_time is not None:
        cropped_df = df[(df.index >= start_time) & (df.index <= end_time)]
    else:
        cropped_df = copy.deepcopy(df)
    
    # Define the start and end times
    start_time = cropped_df.index.min()
    end_time = cropped_df.index.max()
    if is_adjust_time:
        # Create a new index with 10-minute intervals
        new_index = pd.date_range(start=start_time, end=end_time, freq='20T')
        
        # Initialize a new DataFrame with the same columns
        new_df = pd.DataFrame(index=new_index, columns=df.columns)
        
        # Loop through the new index
        for time in tqdm(new_index):
            if time in df.index:
                # Handle multiple rows scenario
                data_at_time = df.loc[time]
                if isinstance(data_at_time, pd.DataFrame):
                    # If multiple rows, you can take the first one, last one, mean, etc.
                    data_at_time = data_at_time.iloc[0]  # Example: take the first occurrence
                new_df.loc[time] = data_at_time
            else:
                # Find the closest timestamps before and after the current timestamp
                before = df.index[df.index < time].max() if any(df.index < time) else None
                after = df.index[df.index > time].min() if any(df.index > time) else None

                # If both a 'before' and 'after' time are found, interpolate
                if pd.notna(before) and pd.notna(after):
                    weight = (time - before) / (after - before)
                    interpolated_values = df.loc[before] + (df.loc[after] - df.loc[before]) * weight
                    new_df.loc[time] = interpolated_values
    else:
        new_df = copy.deepcopy(cropped_df)
    
    if is_convert_time_zone:
        new_df.index = new_df.index.tz_localize('UTC', ambiguous='infer')
        new_df.index = new_df.index.tz_convert('America/Chicago')
        new_df.index = new_df.index.tz_localize(None)
    
    if is_convert_from_knot:
        new_df['sknt'] = new_df['sknt'] * 0.51444
        new_df['gust'] = new_df['gust'] * 0.51444
    
    if is_convert_from_iiHg:
        new_df['alti'] = new_df['alti'] * 33.8639
    
    return new_df

def read_wind_data_V3(folder_path, prefix, suffix, 
                             convert_time_zone=True, 
                             is_convert_from_knot=True, 
                             is_convert_from_iiHg=True, 
                             is_linear_interpolate=True):
    # Initialize an empty list to store the dataframes
    df_list = []
    
    # Iterate over all files in the folder that match the prefix and suffix
    for filename in os.listdir(folder_path):
        if filename.startswith(prefix) and filename.endswith(suffix):
            file_path = os.path.join(folder_path, filename)
            
            # Read the file into a dataframe with automatic header detection
            df = pd.read_csv(file_path)
            
            # Extract the last column name
            last_column = df.columns[-1]
            
            # Keep only the 'valid(UTC)' and last column
            df = df[['valid(UTC)', last_column]]
            
            # Rename the last column to make it unique by appending the filename (or part of it)
            df.rename(columns={last_column: last_column + "_" + filename.split('.')[0]}, inplace=True)
            
            # Convert the 'valid(UTC)' column to datetime format
            df["valid(UTC)"] = pd.to_datetime(df["valid(UTC)"], utc=True)
            
            # Convert to US Central Time (without timezone information) if the option is set to True
            if convert_time_zone:
                df["valid(UTC)"] = df["valid(UTC)"].dt.tz_convert('US/Central').dt.strftime('%Y-%m-%d %H:%M:%S')
            else:
                df["valid(UTC)"] = df["valid(UTC)"].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # Rename 'valid(UTC)' to 'valid'
            df.rename(columns={"valid(UTC)": "valid"}, inplace=True)
            
            # Replace 'M' with np.nan and convert the column to float
            df[df.columns[-1]] = df[df.columns[-1]].replace('M', np.nan).astype(float)
            
            # Check if the column name contains '_sknt_' and convert knots to m/s
            if is_convert_from_knot and '_sknt_' in last_column:
                df[df.columns[-1]] = df[df.columns[-1]] * 0.514444
            
            # Check if the column name contains 'pres' and convert inHg to hPa
            if is_convert_from_iiHg and 'pres' in last_column:
                df[df.columns[-1]] = df[df.columns[-1]] * 33.8639
            
            # Check for 'M' values and apply linear interpolation if necessary
            if is_linear_interpolate and ('drct' in last_column or 'sknt' in last_column):
                df[df.columns[-1]].interpolate(method='linear', inplace=True)
            
            # Set 'valid' as the index for the dataframe
            df.set_index("valid", inplace=True)
            
            # Append the dataframe to the list
            df_list.append(df)
    
    # Merge all dataframes on the index ('valid') using an outer join
    if df_list:
        merged_df = pd.concat(df_list, axis=1)
        
        # Sort by the index, which is 'valid'
        merged_df.sort_index(inplace=True)
        
        return merged_df
    else:
        print("No files found with the specified prefix and suffix.")
        return pd.DataFrame()

def linear_regress_columns(df, predictor_columns, target_column):
    # Filter out rows where the target column or any predictor column is missing
    train_data = df.dropna(subset=[target_column] + predictor_columns)

    # Initialize and train the linear regression model
    model = LinearRegression()
    model.fit(train_data[predictor_columns], train_data[target_column])

    # Find rows where the target column is missing but predictor columns are not
    missing_data = df[df[target_column].isna() & df[predictor_columns].notna().all(axis=1)]

    # Predict missing values
    predicted_values = model.predict(missing_data[predictor_columns])

    # Fill missing values in the DataFrame
    df.loc[missing_data.index, target_column] = predicted_values
    return df