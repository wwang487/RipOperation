from .RipBoxes import RipBoxes
from .RipBox import RipBox
from .RipFrame import RipFrame
from ..SideFunc.dataInteraction import compute_weight, interpolate_vals
from ..DateTime.BasicDateTime import find_closest_timestamps, is_time_ahead
from ..Wave.wave_processing import weighted_average_angle


def track_rip_boxes(rip_dict):
    keys = list(rip_dict.keys())
    keys.sort()
    clean_complete_boxes = []
    for k in range(len(keys)):
        # print(k)
        temp_content = rip_dict.get(keys[k])
        rip_frame = RipFrame(str(keys[k]), temp_content.get('bboxes'), temp_content.get('pvals'))
        if k == 0:
            active_rip_boxes = rip_frame.initialize_rip_boxes()
        else:
            active_rip_boxes, temp_clean_complete_boxes = rip_frame.update_active_rip_boxes(active_rip_boxes)
            clean_complete_boxes += temp_clean_complete_boxes
            active_rip_boxes, temp_clean_complete_boxes = split_merging(active_rip_boxes)
            clean_complete_boxes += temp_clean_complete_boxes
    return clean_complete_boxes

def split_merging(active_rip_boxes):
    check_num = len(active_rip_boxes)
    clean_active_boxes, clean_complete_boxes = [], []
    to_clean = []
    for i in range(check_num):
        if active_rip_boxes[i].get_frame_num() == 1 or i in to_clean:
            continue
        else:
            last_check_id = active_rip_boxes[i].get_last_box().get_unique_id()
            last_iou = active_rip_boxes[i].get_last_iou()
            for j in range(i + 1, check_num):
                if active_rip_boxes[j].get_frame_num() == 1 or j in to_clean:
                    continue
                if active_rip_boxes[j].get_last_box().get_unique_id() == last_check_id:
                    temp_iou = active_rip_boxes[j].get_last_iou()
                    if temp_iou > last_iou:
                        to_clean.append(j)
                    else:
                        to_clean.append(i)
    for i in range(check_num):
        if i not in to_clean:
            clean_active_boxes.append(active_rip_boxes[i])
        else:
            temp_boxes = active_rip_boxes[i]
            temp_boxes.pop_last_box()
            clean_complete_boxes.append(temp_boxes)
    return clean_active_boxes, clean_complete_boxes

def filter_main_boxes(final_boxes):
    res_boxes = []
    for f_b in final_boxes:
        if f_b.get_is_main():
            res_boxes.append(f_b)
    return res_boxes

def eliminate_short_rips(final_boxes, thresh = 2):
    res_boxes = []
    for f_b in final_boxes:
        if f_b.get_frame_num() > thresh:
            res_boxes.append(f_b)
    return res_boxes

def connect_active_boxes(active_rip_boxes):
    pass       

def merge_rip_dict_with_other_data(daily_event_list, other_dict, other_interval, other_keys, 
                                   start_time_str = None, range_limited = True):
    daily_num = len(daily_event_list)
    res = []
    for i in range(daily_num):
        temp_event = daily_event_list[i]
        temp_event_start_time = temp_event.get('start_time')
        temp_event_end_time = temp_event.get('end_time')
        temp_event_peak_time = temp_event.get('peak_time')
        start_low_timestamp, start_high_timestamp = find_closest_timestamps(temp_event_start_time, other_interval, start_time_str = start_time_str)
        end_low_timestamp, end_high_timestamp = find_closest_timestamps(temp_event_end_time, other_interval, start_time_str = start_time_str)
        peak_low_timestamp, peak_high_timestamp = find_closest_timestamps(temp_event_peak_time, other_interval, start_time_str = start_time_str)
        start_weight = compute_weight(temp_event_start_time, start_low_timestamp, start_high_timestamp)
        end_weight = compute_weight(temp_event_end_time, end_low_timestamp, end_high_timestamp)
        peak_weight = compute_weight(temp_event_peak_time, peak_low_timestamp, peak_high_timestamp)
        for k in other_keys:
            if k == 'direction' or k == 'Direction' or k == 'Angle' or k == 'angle':
                temp_event['Start_' + k] = weighted_average_angle(other_dict[start_low_timestamp][k], \
                                            other_dict[start_high_timestamp][k], start_weight, range_limited = range_limited)
                temp_event['End_' + k] = weighted_average_angle(other_dict[end_low_timestamp][k], \
                                            other_dict[end_high_timestamp][k], end_weight, range_limited = range_limited)
                temp_event['Peak_' + k] = weighted_average_angle(other_dict[peak_low_timestamp][k], \
                                            other_dict[peak_high_timestamp][k], peak_weight, range_limited = range_limited)
            else:
                temp_event['Start_' + k] = interpolate_vals(other_dict[start_low_timestamp][k], other_dict[start_high_timestamp][k], start_weight)
                temp_event['End_' + k] = interpolate_vals(other_dict[end_low_timestamp][k], other_dict[end_high_timestamp][k], end_weight)
                temp_event['Peak_' + k] = interpolate_vals(other_dict[peak_low_timestamp][k], other_dict[peak_high_timestamp][k], peak_weight)
        res.append(temp_event)
    return res

def get_max_box_list(final_res, return_type = 'x', max_dist_thresh = None, min_x_thresh = None, max_x_thresh=None):
    # if return_type is 'x', then return the x position of the max box
    # if return_type is 'y', then return the y position of the max box
    # if return_type is 'both', then return both x and y position of the max box
    res_x, res_y = [], []
    for i in range(len(final_res)):
        # max_dist = final_res[i].get('max_dist')
        # x_position = final_res[i].get('max_x_center')
        
        
        temp_boxes =  final_res[i].get('max_boxes')
        for t_b in temp_boxes:
            temp_x, temp_y = (t_b[0] + t_b[2])/2, (t_b[1] + t_b[3])/2
            if max_dist_thresh is not None and temp_y < max_dist_thresh:
                continue
            if min_x_thresh is not None and max_x_thresh is not None:
                if min_x_thresh > temp_x or temp_x > max_x_thresh:
                    continue
            res_x.append(temp_x)
            res_y.append(temp_y)
    if return_type == 'x':
        return res_x
    elif return_type == 'y':
        return res_y
    else:
        return res_x, res_y

def eliminate_left_events(rip_event_list, xcenter_thresh):
    res = []
    for r in rip_event_list:
        if r.get('max_x_center') > xcenter_thresh:
            res.append(r)
    return res

def eliminate_right_events(rip_event_list, xcenter_thresh):
    res = []
    for r in rip_event_list:
        if r.get('max_x_center') < xcenter_thresh:
            res.append(r)
    return res

def eliminate_xposition_events(rip_event_list, xcenter_thresh, is_left = True):
    if is_left:
        return eliminate_left_events(rip_event_list, xcenter_thresh)
    else:
        return eliminate_right_events(rip_event_list, xcenter_thresh)

def eliminate_small_dist_events(rip_event_list, peak_dist_thresh):
    res = []
    for r in rip_event_list:
        if r.get('max_dist') > peak_dist_thresh:
            res.append(r)
    return res

def keep_daytime_events(rip_event_list, start_datetime, end_datetime):
    # I want to keep rip events which start time is after the start datetime but before the end datetime
    res = []
    for r in rip_event_list:
        temp_date = '-'.join(r.get('start_time').split('-')[:3])
        if is_time_ahead(r.get('start_time'), temp_date + '-' + start_datetime) and not is_time_ahead(r.get('start_time'), temp_date + '-' + end_datetime):
            res.append(r)
    return res

def add_classification_col(rip_event_list, angle_thresh_1, angle_thresh_2, height_thresh):
    res = []
    for r in rip_event_list:
        if r.get('Start_height') < height_thresh:
            label = 1
        else:
            if r.get('Start_direction') >= angle_thresh_1 and r.get('Start_direction') <= angle_thresh_2:
                label = 2
            elif r.get('Start_direction') >= 0 and r.get('Start_direction') <= 180:
                label = 3
            else:
                label = 1
        r['Classification'] = label
        res.append(r)
    return res

def add_classification_col_V2(rip_event_list, angle_thresh_1, angle_thresh_2, key = 'breaking_dist', key_thresh = 10):
    res = []
    for r in rip_event_list:
        if r.get(key) < key_thresh:
            label = 1
        else:
            if angle_thresh_1 <= r.get('nearshore_angle') <= angle_thresh_2:
                label = 2
            else:
                label = 3
        r['Classification_V2'] = label
        res.append(r)
    return res

def update_classification(df, angle_1 = -30, angle_2 = 30, thresh=0.3, ratio = 0.3014):
    res = []
    for r in df:
        if r.get('Start_Crossed_WSL') > 0.3:
            if r.get('nearshore_height') < ratio * r.get('Start_Crossed_WSL'):
                if r.get('nearshore_height') < thresh or r.get('breaking_dist') < 0:
                    label = 1
                else:
                    if angle_1 <= r.get('nearshore_angle') <= angle_2:
                        label = 2
                    else:
                        label = 3
            else:
                if angle_1 <= r.get('nearshore_angle') <= angle_2:
                    label = 2
                elif r.get('breaking_dist') >= 0:
                    label = 3
                else:
                    label = 1
        else:
            if r.get('nearshore_height') > 0.1 and r.get('breaking_dist') >= 0:
                if angle_1 <= r.get('nearshore_angle') <= angle_2:
                    label = 2
                else:
                    label = 3
            else:
                label = 4
        r['Classification_V3'] = label
        res.append(r)
    return res

def eliminate_long_duration_events(rip_event_list, max_duration):
    res = []
    for r in rip_event_list:
        if r.get('duration') <= max_duration:
            res.append(r)
    return res