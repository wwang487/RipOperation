import copy as cp
import datetime
from typing import Union

import netCDF4
import numpy as np
import pandas as pd
import utm
import datetime

class TimeDomain:
    def __init__(
        self,
        start_date: datetime.datetime,
        end_date: datetime.datetime,
        interval_delta: pd.Timedelta = pd.to_timedelta(1, unit='hour')
    ) -> None:
        """

        :param start_date: datetime object for start date
        :param end_date: datetime object for end time
        :param interval_delta: datedelta object for interval of time
        """
        self.start_date = start_date
        self.end_date = end_date
        self.date_interval = interval_delta
        self.series_length = -int((start_date - end_date) / interval_delta) + 1
        self.time_series = np.array(self._get_time_series())
        self.year_series = np.linspace(
            self.time_series[0].year,
            self.time_series[-1].year,
            (self.time_series[-1].year - self.time_series[0].year + 1),
            dtype=np.int32)

    def _get_time_series(self) -> list:
        time_series = []
        for i in range(self.series_length):
            time_series.append(self.start_date + i * self.date_interval)
        return time_series

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n < len(self.year_series):
            cur_year = self.year_series[self.n]
            start = datetime.datetime(year=cur_year, month=1, day=1, hour=0)
            end = datetime.datetime(year=cur_year + 1, month=1, day=1, hour=0)
            filter = np.logical_and(self.time_series >= start,
                                    self.time_series < end)
            self.n = self.n + 1
            return filter, self.year_series[self.n - 1]
        else:
            raise StopIteration

    def __len__(self):
        return self.series_length


class Station:
    def __init__(self, time_domain: TimeDomain) -> None:
        self.time_domain = time_domain
        self.time_series = time_domain.time_series
        self.sampling_interval = time_domain.date_interval
        self.x_cord = None
        self.y_cord = None

    def _extract_filter(self,
                        time_series_raw: Union[np.array, list]) -> np.array:
        time_series = self.time_series
        if type(time_series_raw) is list:
            time_series_raw = np.array(time_series_raw)
        time_filter = np.logical_and(time_series_raw >= time_series[0],
                                     time_series_raw <= time_series[-1])
        return time_filter

    def _extract_series(self, time_series_raw: Union[np.array, list],
                        data_series_raw: np.array) -> np.array:
        time_filter = self._extract_filter(time_series_raw=time_series_raw)
        if sum(time_filter) != len(self.time_series):
            time_series1 = time_series_raw[time_filter]
            time_series2 = cp.deepcopy(time_series1[1:])
            time_sc = time_series2 - time_series1[:len(time_series1) - 1]
            time_sc = time_sc / time_sc[0]
            is_one = []
            not_one = []
            init = 0
            for i in range(len(time_sc)):
                if time_sc[i] > 1.0:
                    is_one.append(init)
                    not_one.append(int(time_sc[i]))
                    init = 0
                else:
                    init = init + 1
            is_one.append(init)

            data_series = np.empty([len(self.time_series)])
            data_series[:] = np.NaN

            start_data = 0
            end_data = 0
            start_none = 0
            end_none = -1
            init = 0
            while len(is_one) != 0 and len(not_one) != 0:
                # data part
                num_one = is_one.pop() - 1
                start_data = end_none + 1
                end_data = start_data + num_one
                data_series[start_data:end_data] = data_series_raw[init:init +
                                                                   num_one]
                # None part
                num_not_one = not_one.pop() - 1
                start_none = end_data + 1
                end_none = start_none + num_not_one
                data_series[start_none:end_none] = np.NaN

                init = init + num_one

            if len(is_one) != 0:
                num_one = is_one.pop() - 1
                start_data = end_none + 1
                end_data = start_data + num_one
                data_series[start_data:end_data] = data_series_raw[init:init +
                                                                   num_one]

            if len(not_one) != 0:
                num_not_one = not_one.pop() - 1
                start_none = end_data + 1
                end_none = start_none + num_not_one
                data_series[start_none:end_none] = np.NaN

        else:
            data_series = data_series_raw[time_filter]
        return data_series

    def __len__(self):
        return len(self.time_domain)


class WaveInformationStudyONL(Station):
    def __init__(self,
                 time_domain: TimeDomain,
                 wis_raw: pd.DataFrame) -> None:
        super().__init__(time_domain=time_domain)
        self.wis_raw = wis_raw
        self.wis_id = self._extract_id()

        # extract time series
        self._time_series_raw = self._extract_time_raw()

        # extract wave height/period/direction series
        self.hs_series = self._extract_hs()
        self.period_series = self._extract_period()
        self.angle_series = self._extract_angle()
        self.hs_wind_series = self._extract_series_bycolumn(17)
        self.hs_swell_series = self._extract_series_bycolumn(25)
        self.period_wind_series = self._extract_series_bycolumn(20)
        self.period_swell_series = self._extract_series_bycolumn(28)
        self.angle_wind_series = self._extract_series_bycolumn(23)
        self.angle_swell_series = self._extract_series_bycolumn(31)

        # wind
        self.wind_spd_series = self._extract_series_bycolumn(4)
        self.wind_dir_series = self._extract_series_bycolumn(5)

        # extract the x and y coordinates
        self.lon = self.wis_raw[3][0]
        self.lat = self.wis_raw[2][0]

        self.x_cord, self.y_cord, _, _ = utm.from_latlon(latitude=self.lat,
                                                         longitude=self.lon)

    def _extract_id(self) -> int:
        return int(self.wis_raw[1][0])

    def _extract_time_raw(self) -> pd.DataFrame:
        time_raw = self.wis_raw[0].astype(dtype=str)

        time_series = time_raw.apply(
            lambda x: datetime.datetime(year=int(x[0:4]),
                                        month=int(x[4:6]),
                                        day=int(x[6:8]),
                                        hour=int(x[8:10]),
                                        minute=int(x[10:12]),
                                        second=int(x[12:14])))

        return time_series

    def _extract_hs(self) -> pd.DataFrame:
        return self._extract_series(time_series_raw=self._time_series_raw,
                                    data_series_raw=self.wis_raw[9])

    def _extract_period(self) -> pd.DataFrame:
        return self._extract_series(time_series_raw=self._time_series_raw,
                                    data_series_raw=self.wis_raw[12])

    def _extract_angle(self) -> pd.DataFrame:
        return self._extract_series(time_series_raw=self._time_series_raw,
                                    data_series_raw=self.wis_raw[15])

    def _extract_series_bycolumn(self, col: int) -> pd.DataFrame:
        return self._extract_series(time_series_raw=self._time_series_raw,
                                    data_series_raw=self.wis_raw[col])


class WaveInformationStudy(Station):
    def __init__(self,
                 wis_raw: netCDF4.Dataset,
                 time_domain: TimeDomain) -> None:
        # basic class
        super().__init__(time_domain=time_domain)
        self.wis_raw = wis_raw
        self.wis_id = self._extract_id()

        # extract time series
        self._time_series_raw = self._extract_time_raw()

        # extract wave height/period/direction series
        self.hs_series = self._extract_hs()
        self.period_series = self._extract_period()
        self.angle_series = self._extract_angle()

        # extract the x and y coordinates
        self.lon = self.wis_raw['longitude'][0]
        self.lat = self.wis_raw['latitude'][0]

        self.x_cord, self.y_cord, _, _ = utm.from_latlon(latitude=self.lat,
                                                         longitude=self.lon)

    def _extract_id(self) -> int:
        return int(self.wis_raw['station_name'][0])

    def _extract_time_raw(self) -> datetime.datetime:
        time_num = self.wis_raw['time'][:]
        time_delta = pd.to_timedelta(time_num, unit='s')
        time_series = datetime.datetime(
            year=1970, month=1, day=1, hour=0, second=0) + time_delta

        return time_series

    def _extract_hs(self) -> list:
        return self._extract_series(time_series_raw=self._time_series_raw,
                                    data_series_raw=self.wis_raw['waveHs'][:])

    def _extract_period(self) -> list:
        return self._extract_series(time_series_raw=self._time_series_raw,
                                    data_series_raw=self.wis_raw['waveTp'][:])

    def _extract_angle(self) -> list:
        return self._extract_series(
            time_series_raw=self._time_series_raw,
            data_series_raw=self.wis_raw['waveMeanDirection'][:])


class NOAA(Station):
    def __init__(self,
                 noaa_raw: pd.DataFrame,
                 time_domain: TimeDomain) -> None:
        # basic class
        super().__init__(time_domain=time_domain)
        self.noaa_raw = noaa_raw

        # time series
        self._time_series_raw = self._get_time_raw()
        self.level_series = self._get_level()  # feet2meter
        self.sampling_interval = self.time_series[1] - self.time_series[0]

        # convert string to number
        self._str2num()

    def _get_time_raw(self) -> list:
        dates_series = []
        for i in range(len(self.noaa_raw.iloc[:, 0])):
            item = self.noaa_raw.iloc[i, 0]
            str_splits = item.split('/')
            month = str_splits[0]
            day = str_splits[1]
            year = str_splits[2]
            hour = self.noaa_raw.iloc[i, 1].split(':')[0]
            dates_series.append(
                datetime.datetime(year=int(year),
                                  month=int(month),
                                  day=int(day),
                                  hour=int(hour)))
        return dates_series

    def _get_level(self) -> list:
        return self._extract_series(time_series_raw=self._time_series_raw,
                                    data_series_raw=self.noaa_raw.iloc[:, 4])

    def _str2num(self) -> None:
        for i, item in enumerate(self.level_series):
            if type(item) is str:
                try:
                    self.level_series.iloc[i] = float(item)
                except Exception:
                    self.level_series.iloc[i] = -999

