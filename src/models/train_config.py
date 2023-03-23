USER_COL = [
    "user_id",
    "age",
    "income",
    "sex",
    "kids_flg",
    "user_watch_cnt_all",
    "user_watch_cnt_last_14",
]
ITEM_COL = [
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
CAT_COL = ["age", "income", "sex", "content_type"]

ITEM_STATS_COL = [
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

NUM_NEGATIVES = 3

CATBOOST_PARAMS = {
    "subsample": 0.97,
    "max_depth": 9,
    "n_estimators": 2000,
    "learning_rate": 0.03,
    "scale_pos_weight": NUM_NEGATIVES,
    "l2_leaf_reg": 27,
    "thread_count": -1,
    "verbose": 200,
    "task_type": "CPU",
    # "task_type": "GPU",
    # "devices": "0:1",
    # "bootstrap_type": "Poisson",
}
