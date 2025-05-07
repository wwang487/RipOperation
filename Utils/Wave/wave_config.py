class wave_config:
    def __init__(self, start_year, start_month, start_day, end_year, end_month, end_day, folder, station_name):
        self.start_year = start_year
        self.start_month = start_month
        self.start_day = start_day
        self.end_year = end_year
        self.end_month = end_month
        self.end_day = end_day
        self.folder = folder
        self.station_name = station_name
        self.file = str(self.station_name) + '.nc'
