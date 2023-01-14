"""Tools for recommender systems tasks"""

import pandas as pd
import numpy as np
from more_itertools import pairwise
import scipy.sparse as sp
from itertools import islice, cycle


def compute_metrics(test, recs, top_N):
    """Computes main metrics for recommendations"""
    result = {}
    test_recs = test.set_index(["user_id", "item_id"]).join(
        recs.set_index(["user_id", "item_id"])
    )
    test_recs = test_recs.sort_values(by=["user_id", "rank"])
    test_recs["users_item_count"] = test_recs.groupby(level="user_id")[
        "rank"
    ].transform(np.size)
    test_recs["reciprocal_rank"] = (1 / test_recs["rank"]).fillna(0)
    test_recs["cumulative_rank"] = test_recs.groupby(level="user_id").cumcount() + 1
    test_recs["cumulative_rank"] = test_recs["cumulative_rank"] / test_recs["rank"]
    users_count = test_recs.index.get_level_values("user_id").nunique()
    for k in [top_N // 2, top_N]:
        hit_k = f"hit@{k}"
        test_recs[hit_k] = test_recs["rank"] <= k
        result[f"Precision@{k}"] = (test_recs[hit_k] / k).sum() / users_count
        result[f"Recall@{k}"] = (
            test_recs[hit_k] / test_recs["users_item_count"]
        ).sum() / users_count
    result[f"MAP@{top_N}"] = (
        test_recs["cumulative_rank"] / test_recs["users_item_count"]
    ).sum() / users_count
    result["User_count"] = test["user_id"].nunique()
    result["Leaderboard_all"] = np.nan
    return pd.Series(result)


def stratify(train, filter_dur=300, loyal_treshold=15):
    """
    Calculates watches_items_count per user filtering durations
    less then filter_dur     and calculates strata belonging
    (0 = zero items watched, 2 = more than loyal_treshold items watched,
    1 = inbetween)
    Returns Series id -> strata
    """
    train_copy = train.set_index(["user_id", "item_id"]).sort_values(by="user_id")
    counting = (
        train_copy[train_copy["total_dur"] > filter_dur]
        .groupby(level="user_id")["watched_pct"]
        .count()
    )
    counting = counting.to_frame(name="watched_items")
    counting["strata"] = (counting["watched_items"] >= 1) + 0
    counting["strata"] = (counting["watched_items"] >= loyal_treshold) + counting[
        "strata"
    ]
    return counting["strata"]


class TimeRangeSplit:
    """
    https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.date_range.html
    """

    def __init__(
        self,
        start_date,
        end_date=None,
        freq="D",
        periods=None,
        tz=None,
        normalize=False,
        closed=None,
        train_min_date=None,
        filter_cold_users=True,
        filter_cold_items=True,
        filter_already_seen=True,
        yield_known_items=False,
        items_mapping=None,
    ):

        self.start_date = start_date
        if end_date is None and periods is None:
            raise ValueError(
                'Either "end_date" or "periods" must be non-zero, not both at the same time.'
            )

        self.end_date = end_date
        self.freq = freq
        self.periods = periods
        self.tz = tz
        self.normalize = normalize
        self.closed = closed
        self.train_min_date = pd.to_datetime(train_min_date, errors="raise")
        self.filter_cold_users = filter_cold_users
        self.filter_cold_items = filter_cold_items
        self.filter_already_seen = filter_already_seen
        self.yield_known_items = yield_known_items
        self.items_mapping = items_mapping

        self.date_range = pd.date_range(
            start=start_date,
            end=end_date,
            freq=freq,
            periods=periods,
            tz=tz,
            normalize=normalize,
            closed=closed,
        )

        self.max_n_splits = max(0, len(self.date_range) - 1)
        if self.max_n_splits == 0:
            raise ValueError("Provided parametrs set an empty date range.")

    def split(
        self,
        df,
        user_column="user_id",
        item_column="item_id",
        datetime_column="date",
        fold_stats=False,
    ):
        df_datetime = df[datetime_column]
        if self.train_min_date is not None:
            train_min_mask = df_datetime >= self.train_min_date
        else:
            train_min_mask = df_datetime.notnull()

        date_range = self.date_range[
            (self.date_range >= df_datetime.min())
            & (self.date_range <= df_datetime.max())
        ]

        for start, end in pairwise(date_range):
            fold_info = {"Start date": start, "End date": end}
            train_mask = train_min_mask & (df_datetime <= start)
            train_idx = df.index[train_mask]
            if fold_stats:
                fold_info["Train"] = len(train_idx)

            test_mask = (df_datetime > start) & (df_datetime <= end)
            test_idx = df.index[test_mask]

            if self.filter_cold_users:
                new = np.setdiff1d(
                    df.loc[test_idx, user_column].unique(),
                    df.loc[train_idx, user_column].unique(),
                )
                new_idx = df.index[test_mask & df[user_column].isin(new)]
                test_idx = np.setdiff1d(test_idx, new_idx)
                test_mask = df.index.isin(test_idx)
                if fold_stats:
                    fold_info["New users"] = len(new)
                    fold_info["New users interactions"] = len(new_idx)

            if self.filter_cold_items:
                new = np.setdiff1d(
                    df.loc[test_idx, item_column].unique(),
                    df.loc[train_idx, item_column].unique(),
                )
                new_idx = df.index[test_mask & df[item_column].isin(new)]
                test_idx = np.setdiff1d(test_idx, new_idx)
                test_mask = df.index.isin(test_idx)
                if fold_stats:
                    fold_info["New items"] = len(new)
                    fold_info["New items interactions"] = len(new_idx)

            if self.filter_already_seen:
                user_item = [user_column, item_column]
                train_pairs = df.loc[train_idx, user_item].set_index(user_item).index
                test_pairs = df.loc[test_idx, user_item].set_index(user_item).index
                intersection = train_pairs.intersection(test_pairs)
                print(f"Already seen number: {len(intersection)}")
                test_idx = test_idx[~test_pairs.isin(intersection)]
                # test_mask = rd.df.index.isin(test_idx)
                if fold_stats:
                    fold_info["Known interactions"] = len(intersection)

            if self.yield_known_items:
                items_mapped = df.copy()
                if self.items_mapping != None:
                    items_mapped["item_id"] = items_mapped["item_id"].map(
                        self.items_mapping.get
                    )
                known_items = (
                    items_mapped[(items_mapped["consider_known"] == True) & test_mask]
                    .groupby("user_id")["item_id"]
                    .apply(list)
                    .to_dict()
                )

            if fold_stats:
                fold_info["Test"] = len(test_idx)
            if self.yield_known_items:
                yield (train_idx, test_idx, fold_info, known_items)
            else:
                yield (train_idx, test_idx, fold_info)

    def get_n_splits(self, df, datetime_column="date"):
        df_datetime = df[datetime_column]
        if self.train_min_date is not None:
            df_datetime = df_datetime[df_datetime >= self.train_min_date]

        date_range = self.date_range[
            (self.date_range >= df_datetime.min())
            & (self.date_range <= df_datetime.max())
        ]

        return max(0, len(date_range) - 1)


def get_coo_matrix(
    df,
    user_col="user_id",
    item_col="item_id",
    weight_col=None,
    users_mapping={},
    items_mapping={},
):
    if weight_col is None:
        weights = np.ones(len(df), dtype=np.float32)
    else:
        weights = df[weight_col].astype(np.float32)

    interaction_matrix = sp.coo_matrix(
        (
            weights,
            (df[user_col].map(users_mapping.get), df[item_col].map(items_mapping.get)),
        )
    )
    return interaction_matrix


def generate_implicit_recs_mapper(
    model,
    train_matrix,
    top_N,
    user_mapping,
    item_inv_mapping,
    filter_already_liked_items,
    known_items=None,
    filter_items=None,
    return_scores=False,
):
    def _recs_mapper(user):
        user_id = user_mapping[user]
        if filter_items:
            if user in known_items:
                filtering = set(known_items[user]).union(set(filter_items))
            else:
                filtering = filter_items
        else:
            if known_items and user in known_items:
                filtering = known_items[user]
            else:
                filtering = None
        recs = model.recommend(
            user_id,
            train_matrix,
            N=top_N,
            filter_already_liked_items=filter_already_liked_items,
            filter_items=filtering,
        )
        if return_scores:
            return [item_inv_mapping[item] for item, _ in recs], [
                score for _, score in recs
            ]
        else:
            return [item_inv_mapping[item] for item, _ in recs]

    return _recs_mapper


class PopularRecommender:
    """
    Makes recommendations based on popular items
    """

    def __init__(
        self,
        max_K=10,
        days=30,
        item_column="item_id",
        dt_column="date",
        with_filter=False,
    ):
        self.max_K = max_K if not with_filter else 300
        self.days = days
        self.item_column = item_column
        self.dt_column = dt_column
        self.recommendations = []

    def fit(
        self,
        df,
    ):
        min_date = df[self.dt_column].max().normalize() - pd.DateOffset(days=self.days)
        self.recommendations = (
            df.loc[df[self.dt_column] > min_date, self.item_column]
            .value_counts()
            .head(self.max_K)
            .index.values
        )

    def recommend(self, users=None, N=10):
        recs = self.recommendations[:N]
        if users is None:
            return recs
        else:
            return list(islice(cycle([recs]), len(users)))

    def recommend_with_filter(self, train, user_ids, top_K=10):
        user_ids = pd.Series(user_ids)
        watched_users = user_ids[user_ids.isin(train["user_id"])]
        new_users = user_ids[~user_ids.isin(watched_users)]
        full_recs = self.recommendations
        topk_recs = full_recs[:top_K]
        new_recs = pd.DataFrame({"user_id": new_users})
        new_recs["item_id"] = list(islice(cycle([topk_recs]), len(new_users)))
        watched_recs = pd.DataFrame({"user_id": watched_users})
        watched_recs["item_id"] = 0
        known_items = train.groupby("user_id")["item_id"].apply(list).to_dict()
        watched_recs["additional_N"] = watched_recs["user_id"].apply(
            lambda user_id: len(known_items[user_id]) if user_id in known_items else 0
        )
        watched_recs["total_N"] = watched_recs["additional_N"].apply(
            lambda add_N: add_N + top_K
            if add_N + top_K < len(full_recs)
            else len(full_recs)
        )
        watched_recs["total_recs"] = watched_recs["total_N"].apply(
            lambda total_N: full_recs[:total_N]
        )
        filter_func = lambda row: [
            item
            for item in row["total_recs"]
            if item not in known_items[row["user_id"]]
        ][:top_K]
        watched_recs["item_id"] = watched_recs.loc[:, ["total_recs", "user_id"]].apply(
            filter_func, axis=1
        )
        watched_recs = watched_recs[["user_id", "item_id"]]
        return pd.concat([new_recs, watched_recs], axis=0)


def fill_with_popular(recs, pop_model_fitted, interactions_df, top_K=10):
    """
    Fills missing recommendations with Popular Recommender.
    Takes top_K first recommendations if length of recs exceeds top_K
    """
    recs["len"] = recs["item_id"].apply(lambda x: len(x))
    recs_good = recs[recs["len"] >= top_K].copy()
    recs_good.loc[(recs_good["len"] > top_K), "item_id"] = recs_good.loc[
        (recs_good["len"] > 10), "item_id"
    ].apply(lambda x: x[:10])
    recs_bad = recs[recs["len"] < top_K].copy()
    recs_bad["num_popular"] = top_K - recs_bad.len
    idx_for_filling = recs_bad["user_id"].unique()
    filling_recs = pop_model_fitted.recommend_with_filter(
        interactions_df, idx_for_filling, top_K=top_K
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
    recs_bad["item_id"] = recs_bad["item_id"].apply(lambda x: x[:top_K])
    total_recs = pd.concat(
        [recs_good[["user_id", "item_id"]], recs_bad[["user_id", "item_id"]]], axis=0
    )
    return total_recs
