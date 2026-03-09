from collections import defaultdict


class MetricsCollector:
    def __init__(self) -> None:
        self.counters = defaultdict(int)
        self.gauges = {}

    def inc(self, key: str, value: int = 1) -> None:
        self.counters[key] += value

    def set_gauge(self, key: str, value: float) -> None:
        self.gauges[key] = value

    def snapshot(self) -> dict:
        return {"counters": dict(self.counters), "gauges": self.gauges}
