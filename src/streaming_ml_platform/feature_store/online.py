from __future__ import annotations

from collections import defaultdict


class InMemoryOnlineFeatureStore:
    """Minimal online feature store abstraction for real-time inference."""

    def __init__(self) -> None:
        self._user_features: dict[str, dict] = defaultdict(dict)
        self._session_events: dict[str, list[dict]] = defaultdict(list)

    def upsert_user_features(self, user_id: str, features: dict) -> None:
        self._user_features[user_id].update(features)

    def add_session_event(self, session_id: str, event: dict) -> None:
        self._session_events[session_id].append(event)

    def get_user_features(self, user_id: str) -> dict:
        return self._user_features.get(user_id, {})

    def get_session_events(self, session_id: str) -> list[dict]:
        return self._session_events.get(session_id, [])
