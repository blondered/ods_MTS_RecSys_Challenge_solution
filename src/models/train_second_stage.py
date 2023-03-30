import logging
import pickle

import click
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from train_config import (
    CATBOOST_PARAMS,
    ITEM_COL,
    ITEM_STATS_COL,
    NUM_NEGATIVES,
    USER_COL,
    RANDOM_STATE,
)


@click.command()
@click.argument("interactions_input_path", type=click.Path())
@click.argument("users_processed_input_path", type=click.Path())
@click.argument("items_processed_for_train_input_path", type=click.Path())
@click.argument("implicit_scores_for_train_input_path", type=click.Path())
@click.argument("model_output_path", type=click.Path())
@click.argument("x_train_output_path", type=click.Path())
@click.argument("y_train_output_path", type=click.Path())
@click.argument("x_val_output_path", type=click.Path())
@click.argument("y_val_output_path", type=click.Path())
def train_second_stage(
    interactions_input_path: str,
    users_processed_input_path: str,
    items_processed_for_train_input_path: str,
    implicit_scores_for_train_input_path: str,
    model_output_path: str,
    x_train_output_path: str,
    y_train_output_path: str,
    x_val_output_path: str,
    y_val_output_path: str,
) -> None:
    logging.basicConfig(level=logging.INFO)
    logging.info("Training second stage model")
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

    # taking candidates from implicit model and generating positive samples
    candidates["id"] = candidates.index
    pos = candidates.merge(
        boosting_data[["user_id", "item_id"]], on=["user_id", "item_id"], how="inner"
    )
    pos["target"] = 1

    # Generating negative samples
    pos_group = pos.groupby("user_id")["item_id"].count()

    neg = candidates[~candidates["id"].isin(pos["id"])].copy()
    neg_sampling = pd.DataFrame(neg.groupby("user_id")["id"].apply(list)).join(
        pos_group, on="user_id", rsuffix="p", how="right"
    )
    neg_sampling["num_choices"] = np.clip(
        neg_sampling["item_id"] * NUM_NEGATIVES, a_min=0, a_max=25
    )

    np_random = np.random.RandomState(RANDOM_STATE)


    def row_negative_sampling(row):
        return np_random.choice(row["id"], size=row["num_choices"], replace=False)

    neg_sampling["sample_idx"] = neg_sampling.apply(row_negative_sampling, axis=1)
    idx_chosen = neg_sampling["sample_idx"].explode().values
    neg = neg[neg["id"].isin(idx_chosen)]
    neg["target"] = 0

    # Creating training data sample and early stopping data sample
    boost_idx_train = np.intersect1d(boost_idx, pos["user_id"].unique())
    boost_train_users, boost_eval_users = train_test_split(
        boost_idx_train, test_size=0.1, random_state=RANDOM_STATE
    )
    select_col = ["user_id", "item_id", "implicit_score", "target"]
    boost_train = shuffle(
        pd.concat(
            [
                pos[pos["user_id"].isin(boost_train_users)],
                neg[neg["user_id"].isin(boost_train_users)],
            ]
        )[select_col],
        random_state=RANDOM_STATE
    )
    boost_eval = shuffle(
        pd.concat(
            [
                pos[pos["user_id"].isin(boost_eval_users)],
                neg[neg["user_id"].isin(boost_eval_users)],
            ]
        )[select_col],
        random_state=RANDOM_STATE
    )
    cat_col = ["age", "income", "sex", "content_type"]
    train_feat = boost_train.merge(
        users_df[USER_COL], on=["user_id"], how="left"
    ).merge(items_df[ITEM_COL], on=["item_id"], how="left")
    eval_feat = boost_eval.merge(users_df[USER_COL], on=["user_id"], how="left").merge(
        items_df[ITEM_COL], on=["item_id"], how="left"
    )

    item_stats = pd.read_csv(items_processed_for_train_input_path)
    item_stats = item_stats[ITEM_STATS_COL]
    train_feat = train_feat.join(
        item_stats.set_index("item_id"), on="item_id", how="left"
    )
    eval_feat = eval_feat.join(
        item_stats.set_index("item_id"), on="item_id", how="left"
    )
    drop_col = ["user_id", "item_id"]
    target_col = ["target"]
    x_train = train_feat.drop(drop_col + target_col, axis=1)
    y_train = train_feat[target_col]
    x_val = eval_feat.drop(drop_col + target_col, axis=1)
    y_val = eval_feat[target_col]
    x_train.fillna("None", inplace=True)
    x_val.fillna("None", inplace=True)
    x_train[cat_col] = x_train[cat_col].astype("category")
    x_val[cat_col] = x_val[cat_col].astype("category")

    x_train.to_csv(x_train_output_path, index=False)
    y_train.to_csv(y_train_output_path, index=False)
    x_val.to_csv(x_val_output_path, index=False)
    y_val.to_csv(y_val_output_path, index=False)

    boost_model = CatBoostClassifier(**CATBOOST_PARAMS)
    boost_model.fit(
        x_train,
        y_train,
        eval_set=(x_val, y_val),
        early_stopping_rounds=200,
        cat_features=cat_col,
        plot=False,
    )
    with open(model_output_path, "wb") as dump_file:
        pickle.dump(boost_model, dump_file)


if __name__ == "__main__":
    train_second_stage()
