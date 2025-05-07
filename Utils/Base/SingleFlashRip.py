from datetime import datetime
class SingleFlashRip:
    def __init__(self, start_time=None, end_time=None, peak_time=None, x_min_list=None, y_min_list=None,\
                 x_max_list = None, y_max_list = None):
        self.start_time = start_time
        self.end_time = end_time
        self.peak_time = peak_time
        self.x_min_list = x_min_list
        self.y_min_list = y_min_list
        self.x_max_list = x_max_list
        self.y_max_list = y_max_list

    # Example method to add a position
    def add_min_position(self, x_position, y_position):
        self.x_min_list.append(x_position)
        self.y_min_list.append(y_position)
    
    def add_max_position(self, x_position, y_position):
        self.x_max_list.append(x_position)
        self.y_max_list.append(y_position)

    # Example method to get the duration of the event
    def get_duration(self):
        datetime_str1 = self.start_time
        datetime_str2 = self.end_time
        datetime_obj1 = datetime.strptime(datetime_str1, "%Y-%m-%d-%H-%M-%S")
        datetime_obj2 = datetime.strptime(datetime_str2, "%Y-%m-%d-%H-%M-%S")

        # Calculate the time difference
        difference = datetime_obj1 - datetime_obj2

        # You can format the difference as needed, here it's returned as days, seconds, and microseconds
        return difference


    # Add more methods as needed...
