from pydantic import BaseModel


class UserFeature(BaseModel):
    user_id: str
    total_watch_time_7d: float
    avg_completion_ratio: float
    device_diversity: int
    active_days_30d: int


class ItemFeature(BaseModel):
    item_id: str
    recent_popularity: float
    avg_completion_ratio_item: float
    ctr_approx: float
