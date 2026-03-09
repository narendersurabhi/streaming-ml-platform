from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OnlinePerformanceSnapshot:
    ctr: float
    avg_watch_time: float
    conversion_rate: float


class PerformanceMonitor:
    def __init__(self) -> None:
        self.impressions = 0
        self.clicks = 0
        self.watch_time_seconds = 0.0
        self.conversions = 0

    def record_impression(self) -> None:
        self.impressions += 1

    def record_click(self) -> None:
        self.clicks += 1

    def record_watch_time(self, watch_time_seconds: float) -> None:
        self.watch_time_seconds += watch_time_seconds

    def record_conversion(self) -> None:
        self.conversions += 1

    def snapshot(self) -> OnlinePerformanceSnapshot:
        ctr = self.clicks / self.impressions if self.impressions else 0.0
        avg_watch_time = self.watch_time_seconds / self.clicks if self.clicks else 0.0
        conversion_rate = self.conversions / self.impressions if self.impressions else 0.0
        return OnlinePerformanceSnapshot(ctr=ctr, avg_watch_time=avg_watch_time, conversion_rate=conversion_rate)
