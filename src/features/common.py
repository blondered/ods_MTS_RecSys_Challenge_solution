import pandas as pd

DAYS_OFFSET = 14


def get_interactions_for_train(interactions_df: pd.DataFrame) -> pd.DataFrame:
    max_date = interactions_df["last_watch_dt"].max()
    boosting_split_date = max_date - pd.Timedelta(days=DAYS_OFFSET)
    interactions_boost = interactions_df[
        interactions_df["last_watch_dt"] <= boosting_split_date
    ]
    return interactions_boost
