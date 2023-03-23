import logging

import click
import numpy as np
import pandas as pd
from common import get_interactions_for_train


def add_item_watches_stats(interactions_df, item_stats):
    """
    Computes item watches stats for particular interactions date split
    and adds them to item_stats dataframe
    """

    def smooth(series, window_size, smoothing_func):
        """Computes smoothed interactions statistics for item"""
        series = np.array(series)
        ext = np.r_[
            2 * series[0] - series[window_size - 1 :: -1],
            series,
            2 * series[-1] - series[-1:-window_size:-1],
        ]
        weights = smoothing_func(window_size)
        smoothed = np.convolve(weights / weights.sum(), ext, mode="same")
        return smoothed[window_size : -window_size + 1]

    def trend_slope(series, window_size=7, smoothing_func=np.hamming):
        """Computes trend slope for item interactions"""
        smoothed = smooth(series, window_size, smoothing_func)
        return smoothed[-1] - smoothed[-2]

    keep = item_stats.columns
    max_date = interactions_df["last_watch_dt"].max()
    cols = list(range(7))
    for col in cols:
        watches = interactions_df[
            interactions_df["last_watch_dt"] == max_date - pd.Timedelta(days=6 - col)
        ]
        item_stats = item_stats.join(
            watches.groupby("item_id")["user_id"].count(), lsuffix=col
        )
    item_stats.fillna(0, inplace=True)
    new_colnames = ["user_id" + str(i) for i in range(1, 7)] + ["user_id"]

    def trend_slope_to_row(row):
        return trend_slope(row[new_colnames], window_size=7)

    item_stats["trend_slope"] = item_stats.apply(trend_slope_to_row, axis=1)
    item_stats["watched_in_7_days"] = item_stats[new_colnames].apply(sum, axis=1)
    item_stats["watch_ts_quantile_95"] = 0
    item_stats["watch_ts_median"] = 0
    item_stats["watch_ts_std"] = 0
    for item_id in item_stats.index:
        watches = interactions_df[interactions_df["item_id"] == item_id]
        day_of_year = (
            watches["last_watch_dt"].apply(lambda x: x.dayofyear).astype(np.int64)
        )
        item_stats.loc[item_id, "watch_ts_quantile_95"] = day_of_year.quantile(
            q=0.95, interpolation="nearest"
        )
        item_stats.loc[item_id, "watch_ts_median"] = day_of_year.quantile(
            q=0.5, interpolation="nearest"
        )
        item_stats.loc[item_id, "watch_ts_std"] = day_of_year.std()
    item_stats["watch_ts_quantile_95_diff"] = (
        max_date.dayofyear - item_stats["watch_ts_quantile_95"]
    )
    item_stats["watch_ts_median_diff"] = (
        max_date.dayofyear - item_stats["watch_ts_median"]
    )
    watched_all_time = interactions_df.groupby("item_id")["user_id"].count()
    watched_all_time.name = "watched_in_all_time"
    item_stats = item_stats.join(watched_all_time, on="item_id", how="left")
    item_stats.fillna(0, inplace=True)
    added_cols = [
        "trend_slope",
        "watched_in_7_days",
        "watch_ts_quantile_95_diff",
        "watch_ts_median_diff",
        "watch_ts_std",
        "watched_in_all_time",
    ]
    return item_stats[list(keep) + added_cols]


def add_age_stats(interactions, item_stats, users_df):
    """
    Computes watchers age stats for items with particular interactions
    date split and adds them to item_stats dataframe
    """
    item_stats.reset_index(inplace=True)
    interactions = interactions.set_index("user_id").join(
        users_df[["user_id", "sex", "age", "income"]].set_index("user_id")
    )
    interactions.reset_index(inplace=True)
    interactions["age_overall"] = interactions["age"].replace(
        to_replace={
            "age_18_24": "less_35",
            "age_25_34": "less_35",
            "age_35_44": "over_35",
            "age_45_54": "over_35",
            "age_65_inf": "over_35",
            "age_55_64": "over_35",
        }
    )
    age_stats = interactions.groupby("item_id")["age_overall"].value_counts(
        normalize=True
    )
    age_stats = pd.DataFrame(age_stats)
    age_stats.columns = ["value"]
    age_stats.reset_index(inplace=True)
    age_stats.columns = ["item_id", "age_overall", "value"]
    age_stats = age_stats.pivot(
        index="item_id", columns="age_overall", values="value"
    ).drop("age_unknown", axis=1)
    age_stats.fillna(0, inplace=True)
    item_stats = item_stats.set_index("item_id").join(age_stats)
    item_stats[["less_35", "over_35"]] = item_stats[["less_35", "over_35"]].fillna(0)
    item_stats.rename(
        columns={"less_35": "younger_35_fraction", "over_35": "older_35_fraction"},
        inplace=True,
    )
    return item_stats


def add_sex_stats(interactions, item_stats, users_df):
    """
    Computes watchers sex stats for items with particular interactions date split
    and adds them to item_stats dataframe
    """
    item_stats.reset_index(inplace=True)
    interactions = interactions.set_index("user_id").join(
        users_df[["user_id", "sex", "age", "income"]].set_index("user_id")
    )
    interactions.reset_index(inplace=True)
    sex_stats = interactions.groupby("item_id")["sex"].value_counts(normalize=True)
    sex_stats = pd.DataFrame(sex_stats)
    sex_stats.columns = ["value"]
    sex_stats.reset_index(inplace=True)
    sex_stats.columns = ["item_id", "sex", "value"]
    sex_stats = sex_stats.pivot(index="item_id", columns="sex", values="value").drop(
        "sex_unknown", axis=1
    )
    sex_stats.fillna(0, inplace=True)
    item_stats = item_stats.set_index("item_id").join(sex_stats)
    item_stats[["F", "M"]] = item_stats[["F", "M"]].fillna(0)
    item_stats.rename(
        columns={"F": "female_watchers_fraction", "M": "male_watchers_fraction"},
        inplace=True,
    )
    return item_stats


def compute_stats_and_save(
    interactions_df: pd.DataFrame,
    items_df: pd.DataFrame,
    users_df: pd.DataFrame,
    items_output_path: str,
) -> None:
    item_stats = items_df[["item_id"]].set_index("item_id")
    item_stats = add_item_watches_stats(interactions_df, item_stats)
    item_stats.fillna(0, inplace=True)
    item_stats = add_sex_stats(interactions_df, item_stats, users_df)
    item_stats = add_age_stats(interactions_df, item_stats, users_df)
    items_df_with_features = items_df.join(item_stats, on="item_id", how="left")
    items_df_with_features.to_csv(items_output_path, index=True)


@click.command()
@click.argument("interactions_input_path", type=click.Path())
@click.argument("items_input_path", type=click.Path())
@click.argument("users_input_path", type=click.Path())
@click.argument("items_output_path_for_train", type=click.Path())
@click.argument("items_output_path_for_submit", type=click.Path())
def add_item_stats(
    interactions_input_path: str,
    items_input_path: str,
    users_input_path: str,
    items_output_path_for_train: str,
    items_output_path_for_submit: str,
) -> None:
    logging.basicConfig(level=logging.INFO)
    logging.info("Adding item stats")
    # read data
    interactions_df = pd.read_csv(
        interactions_input_path, parse_dates=["last_watch_dt"]
    )
    items_df = pd.read_csv(items_input_path)
    users_df = pd.read_csv(users_input_path)

    # prepare interactions df for boosting train period of time
    interactions_boost = get_interactions_for_train(interactions_df)

    # compute stats and save data
    compute_stats_and_save(
        interactions_df, items_df, users_df, items_output_path_for_submit
    )
    compute_stats_and_save(
        interactions_boost, items_df, users_df, items_output_path_for_train
    )


if __name__ == "__main__":
    add_item_stats()
