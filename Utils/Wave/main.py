import datetime
import netCDF4
from .station1 import TimeDomain, WaveInformationStudy
from .wave_processing import extract_data_of_time_interval, get_all_candidate_time, convert_np_to_pd
from .wave_processing import get_wave_data_column, add_new_wave_data_column, get_binary_columns
from ..Angles.angleProcessing import dms_to_dec, calculate_azimuth
import argparse

def main_wave(args):
    sy, sm, sd = args.start_year, args.start_month, args.start_day
    ey, em, ed = args.end_year, args.end_month, args.end_day
    file_path = args.folder + args.file
    start = datetime.datetime(year = sy, month = sm, day = sd, hour = 1)
    end = datetime.datetime(year = ey, month = em, day = ed, hour = 1)
    time_domain = TimeDomain(start_date = start, end_date = end)
    ds = netCDF4.Dataset(file_path)
    station = WaveInformationStudy(wis_raw=ds, time_domain=time_domain)
    return station

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    
    parser.add_argument('-fd', '--folder', help = 'The folder to nc file', type = str, default = 'F:/ResearchProjects/RIPCODES/Wave_data/')
    parser.add_argument('-fi', '--file', help = 'The file to nc file', type = str, default = 'ST94058.nc')
    
    parser.add_argument('-ipfd', '--img_parent_folder', help = 'The parent folder to where images are saved', type = str,
                        default = 'F:/ResearchProjects/RIPCODES/Ortho/Selected_OneMinute/')
    
    parser.add_argument('-sy', '--start_year', help = 'The start year of testing', type = int, default = 2019)
    parser.add_argument('-sm', '--start_month', help = 'The start month of testing', type = int, default = 5)
    parser.add_argument('-sd', '--start_day', help = 'The start day of testing', type = int, default =1)
    parser.add_argument('-ey', '--end_year', help = 'The end year of testing', type = int, default = 2019)
    parser.add_argument('-em', '--end_month', help = 'The end month of testing', type = int, default = 10)
    parser.add_argument('-ed', '--end_day', help = 'The end day of testing', type = int, default = 31)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    station = main_wave(args)
    hs = station.hs_series
    period = station.period_series
    angle = station.angle_series
    time = station.time_series
    img_parent_folder = args.img_parent_folder
    new_hs, new_time, start_ind, end_ind = extract_data_of_time_interval(\
            time, hs, args.start_year, args.start_month, args.start_day, \
                args.end_year, args.end_month, args.end_day)
    new_angle = angle[start_ind: end_ind + 1]
    time_arr = get_all_candidate_time(img_parent_folder)
    time_df = convert_np_to_pd(time_arr)
    hs_col = get_wave_data_column(time_arr, new_time, new_hs, choice = 'Linear')
    angle_col = get_wave_data_column(time_arr, new_time, new_angle, choice = 'NN')
    wave_df = add_new_wave_data_column(time_df, hs_col, 'hs')
    wave_df = add_new_wave_data_column(wave_df, angle_col, 'angle')
    lat_arr1, lat_arr2, lon_arr1, lon_arr2 = [43, 23, 36], [43, 24, 0], [-87, 51, 50], [-87, 51, 30]
    coord_1, coord_2 = (dms_to_dec(lat_arr1[0], lat_arr1[1], lat_arr1[2]),\
                        dms_to_dec(lon_arr1[0], lon_arr1[1], lon_arr1[2])),\
                       (dms_to_dec(lat_arr2[0], lat_arr2[1], lat_arr2[2]),\
                            dms_to_dec(lon_arr2[0], lon_arr2[1], lon_arr2[2]))
    azimuth = calculate_azimuth(coord_1, coord_2)
    class_col = get_binary_columns(wave_df, azimuth, 20, 0.5)
    wave_df['class'] = class_col
    