from pydantic import BaseModel


class User(BaseModel):
    user_id: str
    age_bucket: str
    region: str
    subscription_tier: str
    preferred_genres: str
    device_type: str
    activity_level: str


class Item(BaseModel):
    item_id: str
    title: str
    genre: str
    release_year: int
    language: str
    popularity_score: float
    duration: int
    maturity_rating: str
    franchise_flag: int


class Event(BaseModel):
    user_id: str
    item_id: str
    event_type: str
    timestamp: str
    session_id: str
    watch_duration: float
    completion_ratio: float
    search_query: str
