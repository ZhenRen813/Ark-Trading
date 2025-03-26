import datetime
from abc import ABC, abstractmethod
from bisect import bisect_right
from collections.abc import Iterator, Sequence
from typing import Final, Optional
from zoneinfo import ZoneInfo

from .symbols import SymbolInfo, TradedSymbolInfo
from .utils import is_sorted


def get_midnight(date: datetime.date) -> datetime.datetime:
    return datetime.datetime.combine(date, datetime.time.min)


def to_timedelta(dt: datetime.datetime) -> datetime.timedelta:
    return dt - datetime.datetime.min


# TODO: add a bool parameter that tells wether the output should have the same
# timezone of the input
class BaseSchedule(ABC):
    @abstractmethod
    def last_valid_time(self, dt: datetime.datetime) -> datetime.datetime:
        ...

    @abstractmethod
    def next_valid_time(self, dt: datetime.datetime) -> datetime.datetime:
        ...


class ScheduleCombinator(BaseSchedule):
    def __init__(self, schedules: Sequence[BaseSchedule]):
        super().__init__()
        self._schedules = schedules

    def last_valid_time(self, dt: datetime.datetime) -> datetime.datetime:
        new_dt = dt

        while True:
            for schedule in self._schedules:
                new_dt = schedule.last_valid_time(new_dt)

            if new_dt == dt:
                return dt

            dt = new_dt

    def next_valid_time(self, dt: datetime.datetime) -> datetime.datetime:
        new_dt = dt

        while True:
            for schedule in self._schedules:
                new_dt = schedule.next_valid_time(new_dt)

            if new_dt == dt:
                return dt

            dt = new_dt


class HolidaySchedule(BaseSchedule):
    def __init__(
        self, start: datetime.datetime, end: datetime.datetime, is_recurring: bool
    ) -> None:
        super().__init__()

        if start.tzinfo is None or start.tzinfo != end.tzinfo:
            raise ValueError()

        if start.date() != end.date():
            raise ValueError()

        if start >= end:
            raise ValueError()

        self._year = start.year
        self._times = (start, end)
        self.timezone = start.tzinfo
        self.is_recurring = is_recurring

    @property
    def year(self) -> int:
        return self._year

    def last_valid_time(self, dt: datetime.datetime) -> datetime.datetime:
        return self._get_valid_time(dt, reverse=True)

    def next_valid_time(self, dt: datetime.datetime) -> datetime.datetime:
        return self._get_valid_time(dt, reverse=False)

    def _get_valid_time(
        self, dt: datetime.datetime, reverse: bool
    ) -> datetime.datetime:
        orig_dt = dt

        dt = dt.astimezone(self.timezone)

        if self.is_recurring:
            try:
                dt = dt.replace(year=self.year)
            except ValueError:
                # NOTE: it's probably february 29th and the recurring holiday was set
                # on a non-leap year so it's safe to assume this it's not a holiday
                return orig_dt

        index = bisect_right(self._times, dt)

        if index % 2 == 0:
            return orig_dt

        if reverse:
            index -= 1

        return self._times[index].replace(year=orig_dt.year).astimezone(orig_dt.tzinfo)


class WeeklySchedule(BaseSchedule):
    ONE_WEEK: Final[datetime.timedelta] = datetime.timedelta(weeks=1)

    def __init__(
        self, timetable: Sequence[datetime.timedelta], timezone: datetime.tzinfo
    ) -> None:
        super().__init__()

        if not timetable:
            raise ValueError()

        if len(timetable) % 2 != 0:
            raise ValueError()

        if not is_sorted(timetable):
            raise ValueError()

        self.timetable = timetable
        self.timezone = timezone

    def next_valid_time(self, dt: datetime.datetime) -> datetime.datetime:
        return self._get_valid_time(dt, reverse=False)

    def last_valid_time(self, dt: datetime.datetime) -> datetime.datetime:
        return self._get_valid_time(dt, reverse=True)

    def _get_valid_time(
        self, dt: datetime.datetime, reverse: bool
    ) -> datetime.datetime:
        orig_dt = dt
        dt = dt.astimezone(self.timezone)

        days_since_last_sunday = (dt.weekday() + 1) % 7
        last_sunday = dt.date() - datetime.timedelta(days=days_since_last_sunday)
        last_sunday_midnight = get_midnight(last_sunday)

        delta_to_lsm = dt - last_sunday_midnight

        index = bisect_right(self.timetable, delta_to_lsm)
        offset: datetime.timedelta = datetime.timedelta.min

        if index % 2 != 0:
            return orig_dt

        if reverse:
            index -= 1
            if index == 0:
                index == len(self.timetable)
                offset -= self.ONE_WEEK
        else:
            index += 1
            if index == len(self.timetable):
                index = 0
                offset += self.ONE_WEEK

        first_trading_time = last_sunday_midnight + offset + self.timetable[index]
        return first_trading_time.astimezone(orig_dt.tzinfo)


class SymbolSchedule(ScheduleCombinator):
    def __init__(self, sym_info: SymbolInfo) -> None:
        super().__init__(
            [
                self._get_weekly_schedule(sym_info),
                *self._get_holiday_schedules(sym_info),
            ]
        )

    @staticmethod
    def _get_holiday_schedules(sym_info: SymbolInfo) -> Iterator[HolidaySchedule]:
        for holiday in sym_info.symbol.holiday:
            tz = ZoneInfo(holiday.scheduleTimeZone)
            epoch = datetime.datetime.fromtimestamp(0, tz)

            start = epoch + datetime.timedelta(
                days=holiday.holidayDate, seconds=holiday.startSecond
            )
            end = epoch + datetime.timedelta(
                days=holiday.holidayDate, seconds=holiday.endSecond
            )
            yield HolidaySchedule(start, end, holiday.isRecurring)

    @staticmethod
    def _get_weekly_schedule(sym_info: SymbolInfo) -> WeeklySchedule:
        timetable: list[datetime.timedelta] = []
        for interval in sym_info.symbol.schedule:
            start = datetime.timedelta(seconds=interval.startSecond)
            end = datetime.timedelta(seconds=interval.endSecond)
            timetable += (start, end)

        timezone = ZoneInfo(sym_info.symbol.scheduleTimeZone)
        return WeeklySchedule(timetable, timezone)


class IntervalSchedule(BaseSchedule):
    def __init__(self, interval: datetime.timedelta) -> None:
        super().__init__()
        self.interval = interval

    def last_valid_time(self, dt: datetime.datetime) -> datetime.datetime:
        return dt - to_timedelta(dt) % self.interval

    def next_valid_time(self, dt: datetime.datetime) -> datetime.datetime:
        return dt + -to_timedelta(dt) % self.interval


# TODO: this needs to listen to the symbol from SymbolList for the symbols that change
# and update consequently
class ExecutionSchedule(BaseSchedule):
    def __init__(
        self,
        # TODO: should this be a set?
        symbols: Sequence[TradedSymbolInfo],
        trading_interval: datetime.timedelta,
        min_delta_to_close: Optional[datetime.timedelta] = None,
    ) -> None:
        super().__init__()
        self._symbols_schedule = ScheduleCombinator(
            [SymbolSchedule(sym_info) for sym_info in symbols]
        )
        self._symbols = symbols
        self._trading_interval_schedule = IntervalSchedule(trading_interval)
        self._trading_interval = trading_interval
        self._min_delta_to_close = (
            min_delta_to_close
            if min_delta_to_close is not None
            else datetime.timedelta.min
        )
        self._offset: datetime.timedelta = datetime.timedelta.min
        self._offset_date: datetime.date = datetime.date.min

    @property
    def traded_symbols(self) -> Sequence[TradedSymbolInfo]:
        return self._symbols

    @property
    def trading_interval(self) -> datetime.timedelta:
        return self._trading_interval

    def last_valid_time(self, dt: datetime.datetime) -> datetime.datetime:
        # TODO: if skip_day_close and the trading time coincides with the
        # day close time,
        ...

    def next_valid_time(self, dt: datetime.datetime) -> datetime.datetime:
        while True:
            open_time = self._symbols_schedule.next_valid_time(dt)

            if open_time == dt:
                open_time = self._symbols_schedule.last_valid_time(dt)
                new_dt = dt
            else:
                new_dt = open_time

            first_trading_time = self._trading_interval_schedule.next_valid_time(
                open_time
            )
            offset = first_trading_time - open_time

            new_dt += offset
            new_dt = self._trading_interval_schedule.next_valid_time(new_dt)
            new_dt -= offset

            shifted_dt = new_dt + self._min_delta_to_close
            if self._symbols_schedule.next_valid_time(shifted_dt) != shifted_dt:
                new_dt += self.trading_interval

            if new_dt == dt:
                return dt

            dt = new_dt
