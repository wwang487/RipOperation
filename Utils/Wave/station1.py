from __future__ import annotations
import datetime
from typing import Union
import copy as cp
import netCDF4
import numpy as np
import pandas as pd
import utm


class TimeDomain:
    def __init__(
        self,
        start_date: datetime.datetime,
        end_date: datetime.datetime,
        interval_delta: str = "H"
    ) -> None:
        """

        :param start_date: datetime object for start date
        :param end_date: datetime object for end time
        :param interval_delta: datedelta object for interval of time
        """
        self.start_date = start_date
        self.end_date = end_date
        self.date_interval = interval_delta
        self.time_series: pd.DatetimeIndex = self._get_time_series()
        self.series_length = len(self.time_series)
        self.year_series = np.linspace(
            self.time_series[0].year,
            self.time_series[-1].year,
            (self.time_series[-1].year - self.time_series[0].year + 1),
            dtype=np.int32)

    def _get_time_series(self) -> pd.DatetimeIndex:
        date_range = pd.date_range(
            start=self.start_date, end=self.end_date, freq=self.date_interval)
        return date_range

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
            filter = (self.time_series >= start) & (self.time_series < end)
            self.n = self.n + 1
            return filter, self.year_series[self.n - 1]
        else:
            raise StopIteration

    def __add__(self, other: TimeDomain) -> TimeDomain:
        if other.date_interval != self.date_interval:
            raise ValueError(
                "cannot concate two series with different interval")

        start_date = min(self.start_date, other.start_date)
        end_date = max(self.end_date, other.end_date)
        time_domain = TimeDomain(start_date=start_date, end_date=end_date,
                                 interval_delta=self.date_interval)
        new_time_series = self.time_series.union(other.time_series)
        time_domain.time_series = pd.DatetimeIndex(new_time_series)
        return time_domain

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
                        time_series_raw: Union[np.ndarray, list]) \
            -> np.ndarray:
        time_series = self.time_series
        if type(time_series_raw) is list:
            time_series_raw = np.array(time_series_raw)
        time_filter = np.logical_and(time_series_raw >= time_series[0],
                                     time_series_raw <= time_series[-1])
        return time_filter

    def _extract_series(self, time_series_raw: Union[np.ndarray,
                                                     pd.DatetimeIndex],
                        data_series_raw: Union[np.ndarray,
                                               pd.Series]) -> pd.Series:
        time_filter = self._extract_filter(time_series_raw=time_series_raw)

        if not isinstance(data_series_raw, pd.Series):
            series = pd.Series(data_series_raw[time_filter],
                               index=time_series_raw[time_filter])
        else:
            series = data_series_raw[time_filter]
            series.index = time_series_raw[time_filter]
        # remove duplicated
        series = series[~series.index.duplicated()]
        # fill the missing data
        series = series.reindex(pd.DatetimeIndex(self.time_series),
                                fill_value=-999)
        return series

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

    def _extract_time_raw(self):
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
                                    data_series_raw=self.wis_raw['waveTm1'][:])

    def _extract_angle(self) -> list:
        return self._extract_series(
            time_series_raw=self._time_series_raw,
            data_series_raw=self.wis_raw['waveMeanDirection'][:])
    
        #    self.hs_wind_series = self._extract_series_bycolumn(17)
        # self.hs_swell_series = self._extract_series_bycolumn(25)
        # self.period_wind_series = self._extract_series_bycolumn(20)
        # self.period_swell_series = self._extract_series_bycolumn(28)
        # self.angle_wind_series = self._extract_series_bycolumn(23)
        # self.angle_swell_series = self._extract_series_bycolumn(31)
    
    def _extract_by_column_name(self, column_name: str) -> list:
        return self._extract_series(time_series_raw=self._time_series_raw,
                                    data_series_raw=self.wis_raw[column_name][:])

class NOAA(Station):
    def __init__(self,
                 noaa_raw: pd.DataFrame,
                 time_domain: TimeDomain) -> None:
        # basic class
        super().__init__(time_domain=time_domain)
        self.noaa_raw = noaa_raw

        # time series
        self._time_series_raw = self._get_time_raw()
        self.level_series: pd.Series = self._get_level()  # feet2meter
        self.sampling_interval = self.time_series[1] - self.time_series[0]

        # convert string to number
        self._str2num()

    def _get_time_raw(self) -> pd.DatetimeIndex:
        df = self.noaa_raw
        df["Date Time"] = pd.to_datetime(df["Date Time"])

        return df['Date Time']

    def _get_level(self) -> pd.Series:
        return self._extract_series(time_series_raw=self._time_series_raw,
                                    data_series_raw=self.noaa_raw.iloc[:, 1])

    def _str2num(self) -> None:
        for i, item in enumerate(self.level_series):
            if type(item) is str:
                try:
                    self.level_series.iloc[i] = float(item)
                except Exception:
                    self.level_series.iloc[i] = -999


class NDBC(Station):
    def __init__(self,
                 data_raw_df: pd.DataFrame,
                 time_domain: TimeDomain) -> None:
        super().__init__(time_domain)
        self.__data_raw_df: pd.DataFrame = data_raw_df
        self._construct_timeseries()
        self.__hs_series = self._extract_series(
            self.__data_raw_df.index, self.__data_raw_df["WVHT"])
        self.__per_series = self._extract_series(
            self.__data_raw_df.index, self.__data_raw_df["DPD"])
        self.__angle_series = self._extract_series(
            self.__data_raw_df.index, self.__data_raw_df["MWD"])

    def __add__(self, other: NDBC):
        new_time_domain = self.time_domain + other.time_domain
        new_ndbc = cp.deepcopy(self)
        new_ndbc.time_domain = new_time_domain
        new_ndbc.time_series = new_time_domain.time_series
        new_ndbc.__hs_series = np.concatenate(
            [self.__hs_series, other.__hs_series])
        return new_ndbc

    def _construct_timeseries(self):
        # check the format of dataframe
        df = self.__data_raw_df
        if df.iloc[0, 0] == "#yr":
            df = df.drop(df.index[0])

        # df = df.applymap(lambda x: float(x) if '.' in x else int(x))
        if 'mm' not in self.__data_raw_df.columns:  # no minute data
            df["datetime"] = pd.to_datetime(df.iloc[:, 0].astype(str) + '-' +
                                            df.iloc[:, 1].astype(str) + '-' +
                                            df.iloc[:, 2].astype(str) + ' ' +
                                            df.iloc[:, 3].astype(str) +
                                            ':00:00')
            df.set_index('datetime', inplace=True)
            df = df.replace(99.0, np.nan)
            df = df.replace(999.0, np.nan)

            df = df.applymap(lambda x: float(x))
        else:  # minute data
            df['datetime'] = pd.to_datetime(df.iloc[:, 0].astype(str) + '-' +
                                            df.iloc[:, 1].astype(str) + '-' +
                                            df.iloc[:, 2].astype(str) + ' ' +
                                            df.iloc[:, 3].astype(str) + ':' +
                                            df.iloc[:, 4].astype(str))
            df.set_index('datetime', inplace=True)
            df = df.applymap(lambda x: float(x))
            df = df.replace(99.0, np.nan)
            df = df.replace(999.0, np.nan)
            # resample the data
            df = df.resample('H').mean()

        self.__data_raw_df = df

    @property
    def hs_series(self):
        return self.__hs_series

    @property
    def per_series(self):
        return self.__per_series

    @property
    def angle_series(self):
        return self.__angle_series
