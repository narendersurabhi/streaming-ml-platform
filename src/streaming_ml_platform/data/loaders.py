from pathlib import Path

import pandas as pd


def load_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def load_raw_data(raw_dir: Path) -> dict[str, pd.DataFrame]:
    return {
        "users": load_table(raw_dir / "users.csv"),
        "items": load_table(raw_dir / "items.csv"),
        "events": load_table(raw_dir / "events.csv"),
    }
