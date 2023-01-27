import pickle
import click
import logging

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


@click.command()
@click.argument("interactions_input_path", type=click.Path())
@click.argument("users_processed_input_path", type=click.Path())
@click.argument("items_processed_for_train_input_path", type=click.Path())
@click.argument("implicit_scores_for_train_input_path", type=click.Path())
@click.argument("model_output_path", type=click.Path())
def train_second_stage(
    interactions_input_path: str, users_processed_input_path: str, items_processed_for_train_input_path: str,
    implicit_scores_for_train_input_path: str, model_output_path: str
) -> None:
    logging.basicConfig(level=logging.INFO)
    logging.info("Start training second stage")
    # reading data
    users_df = pd.read_csv(users_processed_input_path)
    items_df = pd.read_csv(items_processed_for_train_input_path)
    interactions_df = pd.read_csv(
        interactions_input_path, parse_dates=["last_watch_dt"]
    )
    candidates = pd.read_csv(implicit_scores_for_train_input_path)
    last_date_df = interactions_df["last_watch_dt"].max()

    # taking interactions in last 14 days as boosting train
    boosting_split_date = last_date_df - pd.Timedelta(days=14)
    boosting_data = interactions_df[
        (interactions_df["last_watch_dt"] > boosting_split_date)
    ].copy()
    boost_idx = boosting_data["user_id"].unique()

    logging.info("Preparing positive samples")
    # taking candidates from implicit model and generating positive samples
    candidates["id"] = candidates.index
    pos = candidates.merge(
        boosting_data[["user_id", "item_id"]], on=["user_id", "item_id"], how="inner"
    )
    pos["target"] = 1

    logging.info("Preparing negative samples")
    # Generating negative samples
    num_negatives = 3
    pos_group = pos.groupby("user_id")["item_id"].count()

    neg = candidates[~candidates["id"].isin(pos["id"])].copy()
    neg_sampling = pd.DataFrame(neg.groupby("user_id")["id"].apply(list)).join(
        pos_group, on="user_id", rsuffix="p", how="right"
    )
    neg_sampling["num_choices"] = np.clip(
        neg_sampling["item_id"] * num_negatives, a_min=0, a_max=25
    )
    # neg.to_csv("data/interim/neg.csv")
    # pos.to_csv("data/interim/pos.csv")
    # neg_sampling.to_csv("data/interim/neg_sampling.csv")
    func = lambda row: np.random.choice(row["id"], size=row["num_choices"], replace=False)
    neg_sampling["sample_idx"] = neg_sampling.apply(func, axis=1)
    idx_chosen = neg_sampling["sample_idx"].explode().values
    neg = neg[neg["id"].isin(idx_chosen)]
    neg["target"] = 0

    logging.info("Creating train and eval data")
    # Creating training data sample and early stopping data sample
    boost_idx_train = np.intersect1d(boost_idx, pos["user_id"].unique())
    boost_train_users, boost_eval_users = train_test_split(
        boost_idx_train, test_size=0.1, random_state=345
    )
    select_col = ["user_id", "item_id", "implicit_score", "target"]
    boost_train = shuffle(
        pd.concat(
            [
                pos[pos["user_id"].isin(boost_train_users)],
                neg[neg["user_id"].isin(boost_train_users)],
            ]
        )[select_col]
    )
    boost_eval = shuffle(
        pd.concat(
            [
                pos[pos["user_id"].isin(boost_eval_users)],
                neg[neg["user_id"].isin(boost_eval_users)],
            ]
        )[select_col]
    )
    user_col = [
        "user_id",
        "age",
        "income",
        "sex",
        "kids_flg",
        "boost_user_watch_cnt_all",
        "boost_user_watch_cnt_last_14",
    ]
    item_col = [
        "item_id",
        "content_type",
        "countries_max",
        "for_kids",
        "age_rating",
        "studios_max",
        "genres_max",
        "genres_min",
        "genres_med",
        "release_novelty",
    ]
    item_stats_col = [
        "item_id",
        "watched_in_7_days",
        "watch_ts_std",
        "trend_slope",
        "watch_ts_quantile_95_diff",
        "watch_ts_median_diff",
        "watched_in_all_time",
        "male_watchers_fraction",
        "female_watchers_fraction",
        "younger_35_fraction",
        "older_35_fraction",
    ]
    cat_col = ["age", "income", "sex", "content_type"]
    train_feat = boost_train.merge(users_df[user_col], on=["user_id"], how="left").merge(
        items_df[item_col], on=["item_id"], how="left"
    )
    eval_feat = boost_eval.merge(users_df[user_col], on=["user_id"], how="left").merge(
        items_df[item_col], on=["item_id"], how="left"
    )

    logging.info("Collecting item stats")
    item_stats = pd.read_csv(items_processed_for_train_input_path)
    item_stats = item_stats[item_stats_col]
    train_feat = train_feat.join(item_stats.set_index("item_id"), on="item_id", how="left")
    eval_feat = eval_feat.join(item_stats.set_index("item_id"), on="item_id", how="left")
    drop_col = ["user_id", "item_id"]
    target_col = ["target"]
    X_train = train_feat.drop(drop_col + target_col, axis=1)
    y_train = train_feat[target_col]
    X_val = eval_feat.drop(drop_col + target_col, axis=1)
    y_val = eval_feat[target_col]
    X_train.fillna("None", inplace=True)
    X_val.fillna("None", inplace=True)
    X_train[cat_col] = X_train[cat_col].astype("category")
    X_val[cat_col] = X_val[cat_col].astype("category")

    logging.info("Fitting catboost model")
    # Training CatBoost classifier with parameters previously chosen on cross validation
    params = {
        "subsample": 0.97,
        "max_depth": 9,
        "n_estimators": 2000,
        "learning_rate": 0.03,
        "scale_pos_weight": num_negatives,
        "l2_leaf_reg": 27,
        "thread_count": -1,
        "verbose": 200,
        "task_type": "GPU",
        "devices": "0:1",
        "bootstrap_type": "Poisson",
    }
    boost_model = CatBoostClassifier(**params)
    boost_model.fit(
        X_train,
        y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=200,
        cat_features=cat_col,
        plot=False,
    )
    logging.info("Saving catboost model")
    with open(model_output_path, "wb") as f:
        pickle.dump(boost_model, f)


if __name__ == "__main__":
    train_second_stage()
