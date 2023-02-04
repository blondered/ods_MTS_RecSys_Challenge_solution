import logging

import click
import pandas as pd


def add_user_stats(interactions_df, users_df, split_name=""):
    """
    Computes user watches stats for particular interactions date split
    and adds them to users dataframe with specific name
    """
    user_watch_count_all = (
        interactions_df[interactions_df["total_dur"] > 300]
        .groupby(by="user_id")["item_id"]
        .count()
    )
    max_date_df = interactions_df["last_watch_dt"].max()
    user_watch_count_last_14 = (
        interactions_df[
            (interactions_df["total_dur"] > 300)
            & (
                interactions_df["last_watch_dt"]
                >= (max_date_df - pd.Timedelta(days=14))
            )
        ]
        .groupby(by="user_id")["item_id"]
        .count()
    )
    user_watch_count_all.name = split_name + "user_watch_cnt_all"
    user_watch_count_last_14.name = split_name + "user_watch_cnt_last_14"
    user_watches = pd.DataFrame(user_watch_count_all).join(
        user_watch_count_last_14, how="outer"
    )
    user_watches.fillna(0, inplace=True)
    cols = user_watches.columns
    user_watches[cols] = user_watches[cols].astype("int64")
    users_df = users_df.join(user_watches, on="user_id", how="outer")
    users_df[cols] = users_df[cols].fillna(0)
    users_df["age"] = users_df["age"].fillna("age_unknown")
    users_df["income"] = users_df["income"].fillna("income_unknown")
    users_df["sex"] = users_df["sex"].fillna("sex_unknown")
    users_df["kids_flg"] = users_df["kids_flg"].fillna(False)
    return users_df


@click.command()
@click.argument("interactions_input_path", type=click.Path())
@click.argument("users_input_path", type=click.Path())
@click.argument("users_output_path", type=click.Path())
def add_and_save_user_stats(
    interactions_input_path: str,
    users_input_path: str,
    users_output_path: str,
) -> None:
    logging.basicConfig(level=logging.INFO)
    logging.info("Adding user stats")

    interactions_df = pd.read_csv(
        interactions_input_path, parse_dates=["last_watch_dt"]
    )
    users_df = pd.read_csv(users_input_path)

    max_date = interactions_df["last_watch_dt"].max()
    boosting_split_date = max_date - pd.Timedelta(days=14)
    interactions_boost = interactions_df[
        interactions_df["last_watch_dt"] <= boosting_split_date
    ]

    users_df = add_user_stats(interactions_boost, users_df, split_name="boost_")
    users_df = add_user_stats(interactions_df, users_df, split_name="")
    users_df.to_csv(users_output_path, index=True)


if __name__ == "__main__":
    add_and_save_user_stats()
