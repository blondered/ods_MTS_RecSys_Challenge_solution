import logging
import pickle
from itertools import cycle, islice

import click
import pandas as pd
from train_config import CAT_COL, ITEM_COL, USER_COL


def to_string_func(numbers_list):
    """
    Converts list to its string representation
    """
    strings_list = list(map(str, list(numbers_list)))
    return "[" + (", ").join(strings_list) + "]"


class PopularRecommender:
    """
    Makes recommendations based on popular items
    """

    def __init__(
        self,
        max_k=10,
        days=30,
        item_column="item_id",
        dt_column="date",
        with_filter=False,
    ):
        self.max_k = max_k if not with_filter else 300
        self.days = days
        self.item_column = item_column
        self.dt_column = dt_column
        self.recommendations = []

    def fit(
        self,
        interactions_df,
    ):
        min_date = interactions_df[self.dt_column].max().normalize() - pd.DateOffset(
            days=self.days
        )
        self.recommendations = (
            interactions_df.loc[
                interactions_df[self.dt_column] > min_date, self.item_column
            ]
            .value_counts()
            .head(self.max_k)
            .index.values
        )

    def recommend(self, users=None, top_k=10):
        recs = self.recommendations[:top_k]
        if users is None:
            return recs
        return list(islice(cycle([recs]), len(users)))

    def recommend_with_filter(self, train, user_ids, top_k=10):
        user_ids = pd.Series(user_ids)
        watched_users = user_ids[user_ids.isin(train["user_id"])]
        new_users = user_ids[~user_ids.isin(watched_users)]
        full_recs = self.recommendations
        topk_recs = full_recs[:top_k]
        new_recs = pd.DataFrame({"user_id": new_users})
        new_recs["item_id"] = list(islice(cycle([topk_recs]), len(new_users)))
        watched_recs = pd.DataFrame({"user_id": watched_users})
        watched_recs["item_id"] = 0
        known_items = train.groupby("user_id")["item_id"].apply(list).to_dict()
        watched_recs["additional_N"] = watched_recs["user_id"].apply(
            lambda user_id: len(known_items[user_id]) if user_id in known_items else 0
        )
        watched_recs["total_N"] = watched_recs["additional_N"].apply(
            lambda add_N: add_N + top_k
            if add_N + top_k < len(full_recs)
            else len(full_recs)
        )
        watched_recs["total_recs"] = watched_recs["total_N"].apply(
            lambda total_N: full_recs[:total_N]
        )

        def row_filtering_func(row):
            return [
                item
                for item in row["total_recs"]
                if item not in known_items[row["user_id"]]
            ][:top_k]

        watched_recs["item_id"] = watched_recs.loc[:, ["total_recs", "user_id"]].apply(
            row_filtering_func, axis=1
        )
        watched_recs = watched_recs[["user_id", "item_id"]]
        return pd.concat([new_recs, watched_recs], axis=0)


def fill_with_popular(recs, pop_model_fitted, interactions_df, top_k=10):
    """
    Fills missing recommendations with Popular Recommender.
    Takes top_k first recommendations if length of recs exceeds top_k
    """
    recs["len"] = recs["item_id"].apply(lambda x: len(x))
    recs_good = recs[recs["len"] >= top_k].copy()
    recs_good.loc[(recs_good["len"] > top_k), "item_id"] = recs_good.loc[
        (recs_good["len"] > 10), "item_id"
    ].apply(lambda x: x[:10])
    recs_bad = recs[recs["len"] < top_k].copy()
    recs_bad["num_popular"] = top_k - recs_bad.len
    idx_for_filling = recs_bad["user_id"].unique()
    filling_recs = pop_model_fitted.recommend_with_filter(
        interactions_df, idx_for_filling, top_k=top_k
    )
    recs_bad = recs_bad.join(
        filling_recs.set_index("user_id"), on="user_id", how="left", rsuffix="1"
    )
    recs_bad.loc[(recs_bad["len"] > 0), "item_id"] = (
        recs_bad.loc[(recs_bad["len"] > 0), "item_id"]
        + recs_bad.loc[(recs_bad["len"] > 0), "item_id1"]
    )
    recs_bad.loc[(recs_bad["len"] == 0), "item_id"] = recs_bad.loc[
        (recs_bad["len"] == 0), "item_id1"
    ]
    recs_bad["item_id"] = recs_bad["item_id"].apply(lambda x: x[:top_k])
    total_recs = pd.concat(
        [recs_good[["user_id", "item_id"]], recs_bad[["user_id", "item_id"]]], axis=0
    )
    return total_recs


@click.command()
@click.argument("interactions_input_path", type=click.Path())
@click.argument("users_processed_input_path", type=click.Path())
@click.argument("items_processed_for_submit_input_path", type=click.Path())
@click.argument("implicit_scores_for_submit_input_path", type=click.Path())
@click.argument("sample_submission_input_path", type=click.Path())
@click.argument("model_input_path", type=click.Path())
@click.argument("submission_output_path", type=click.Path())
def prepare_submission(
    interactions_input_path: str,
    users_processed_input_path: str,
    items_processed_for_submit_input_path: str,
    implicit_scores_for_submit_input_path: str,
    sample_submission_input_path: str,
    model_input_path: str,
    submission_output_path: str,
) -> None:
    # Reading data
    logging.basicConfig(level=logging.INFO)
    logging.info("Preparing recommendations")
    users_interactions_df = pd.read_csv(users_processed_input_path)
    items_interactions_df = pd.read_csv(items_processed_for_submit_input_path)
    interactions_df = pd.read_csv(
        interactions_input_path, parse_dates=["last_watch_dt"]
    )
    submission = pd.read_csv(sample_submission_input_path)
    candidates = pd.read_csv(
        implicit_scores_for_submit_input_path,
        usecols=["user_id", "item_id", "implicit_score"],
    )

    with open(model_input_path, "rb") as dump_file:
        boost_model = pickle.load(dump_file)

    # Constructing data for predictions

    candidates.dropna(subset=["item_id"], axis=0, inplace=True)
    submit_feat = candidates.merge(
        users_interactions_df[USER_COL], on=["user_id"], how="left"
    ).merge(items_interactions_df[ITEM_COL], on=["item_id"], how="left")
    full_train = submit_feat.fillna("None")
    full_train[CAT_COL] = full_train[CAT_COL].astype("category")
    item_stats = pd.read_csv("data/item_stats_for_submit.csv")
    full_train = full_train.join(
        item_stats.set_index("item_id"), on="item_id", how="left"
    )

    # Renaming columns to match classifier feature names
    cols = ["user_id", "item_id"]
    cols.extend(boost_model.feature_names_)
    cols = cols[:7] + ["user_watch_cnt_all", "user_watch_cnt_last_14"] + cols[9:]
    full_train = full_train[cols]
    full_train_new_names = ["user_id", "item_id"] + boost_model.feature_names_
    full_train.columns = full_train_new_names

    # Making predictions for warm users
    y_pred_all = boost_model.predict_proba(
        full_train.drop(["user_id", "item_id"], axis=1)
    )
    full_train["boost_pred"] = y_pred_all[:, 1]
    full_train = full_train[["user_id", "item_id", "boost_pred"]]
    full_train = full_train.sort_values(
        by=["user_id", "boost_pred"], ascending=[True, False]
    )
    full_train["rank"] = full_train.groupby("user_id").cumcount() + 1
    full_train = full_train[full_train["rank"] <= 10].drop("boost_pred", axis=1)
    full_train["item_id"] = full_train["item_id"].astype("int64")
    boost_recs = full_train.groupby("user_id")["item_id"].apply(list)
    boost_recs = pd.DataFrame(boost_recs)
    boost_recs.reset_index(inplace=True)

    # Making predictions for cold users with Popular Recommender
    idx_for_popular = list(
        set(submission["user_id"].unique()).difference(
            set(boost_recs["user_id"].unique())
        )
    )
    pop_model = PopularRecommender(days=30, dt_column="last_watch_dt", with_filter=True)
    pop_model.fit(interactions_df)
    recs_popular = pop_model.recommend_with_filter(
        interactions_df, idx_for_popular, top_k=10
    )
    all_recs = pd.concat([boost_recs, recs_popular], axis=0)

    # Filling short recommendations woth popular items
    all_recs = fill_with_popular(all_recs, pop_model, interactions_df)

    # Changing recommendations format and saving submission
    all_recs["item_id"] = all_recs["item_id"].apply(to_string_func)
    all_recs.to_csv(submission_output_path, index=False)
    logging.info("All done")


if __name__ == "__main__":
    prepare_submission()
