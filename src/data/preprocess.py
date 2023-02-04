import numpy as np
import pandas as pd
import logging
import click


@click.command()
@click.argument("interactions_input_path", type=click.Path())
@click.argument("items_input_path", type=click.Path())
@click.argument("users_input_path", type=click.Path())
@click.argument("interactions_output_path", type=click.Path())
@click.argument("items_output_path", type=click.Path())
@click.argument("users_output_path", type=click.Path())
def preprocess(
    interactions_input_path: str,
    items_input_path: str,
    users_input_path: str,
    interactions_output_path: str,
    items_output_path: str,
    users_output_path: str,
) -> None:
    logging.basicConfig(level=logging.INFO)
    logging.info("Preprocessing data")

    # Reading files
    users_df = pd.read_csv(users_input_path)
    items_df = pd.read_csv(items_input_path)
    interactions_df = pd.read_csv(
        interactions_input_path, parse_dates=["last_watch_dt"]
    )

    # Interactions preprocessing
    interactions_df["watched_pct"] = (
        interactions_df["watched_pct"].astype(pd.Int8Dtype()).fillna(0)
    )
    interactions_df["last_watch_dt"] = pd.to_datetime(interactions_df["last_watch_dt"])

    # Users info preprocessing
    users_df["age"] = users_df["age"].fillna("age_unknown").astype("category")
    users_df["income"] = users_df["income"].fillna("income_unknown").astype("category")
    users_df.loc[users_df.sex == "М", "sex"] = "M"
    users_df.loc[users_df.sex == "Ж", "sex"] = "F"
    users_df["sex"] = users_df["sex"].fillna("sex_unknown").astype("category")
    users_df["kids_flg"] = users_df["kids_flg"].astype("bool")

    # Items info preprocessing
    items_df["content_type"] = items_df["content_type"].astype("category")
    items_df["title"] = items_df["title"].str.lower()
    items_df["title_orig"] = items_df["title_orig"].fillna("None")
    items_df["for_kids"] = items_df["for_kids"].fillna(0).astype("bool")
    items_df.loc[items_df.age_rating.isna(), "age_rating"] = 0
    items_df["age_rating"] = items_df["age_rating"].astype("category")

    # Release novelty
    items_df.loc[items_df["release_year"] < 1980, "release_novelty"] = 1
    items_df.loc[items_df["release_year"] >= 2020, "release_novelty"] = 6
    novelty = 1
    for i in range(1980, 2020, 10):
        novelty += 1
        items_df.loc[
            (items_df["release_year"] >= i) & (items_df["release_year"] < i + 10),
            "release_novelty",
        ] = novelty
    items_df = items_df.drop(columns=["release_year"])

    # Genres, countries and studios
    items_df["genres_list"] = items_df["genres"].apply(lambda x: x.split(", "))
    num_genres = pd.Series(np.hstack(items_df["genres_list"].values)).value_counts()
    items_df["genres_min"] = items_df["genres_list"].apply(
        lambda x: min([num_genres[el] for el in x])
    )
    items_df["genres_max"] = items_df["genres_list"].apply(
        lambda x: max([num_genres[el] for el in x])
    )
    items_df["genres_med"] = items_df["genres_list"].apply(
        lambda x: (np.median([num_genres[el] for el in x]))
    )
    items_df["countries"].fillna("None", inplace=True)
    items_df["countries"] = items_df["countries"].str.lower()
    items_df["countries_list"] = items_df["countries"].apply(
        lambda x: x.split(", ") if ", " in x else [x]
    )
    num_countries = pd.Series(
        np.hstack(items_df["countries_list"].values)
    ).value_counts()
    items_df["countries_max"] = items_df["countries_list"].apply(
        lambda x: max([num_countries[el] for el in x])
    )
    items_df["studios"].fillna("None", inplace=True)
    items_df["studios"] = items_df["studios"].str.lower()
    items_df["studios_list"] = items_df["studios"].apply(
        lambda x: x.split(", ") if ", " in x else [x]
    )
    num_studios = pd.Series(np.hstack(items_df["studios_list"].values)).value_counts()
    items_df["studios_max"] = items_df["studios_list"].apply(
        lambda x: max([num_studios[el] for el in x])
    )
    items_df.drop(
        ["countries_list", "genres_list", "studios_list"], axis=1, inplace=True
    )

    # Saving preprocessed files
    users_df.to_csv(users_output_path, index=False)
    items_df.to_csv(items_output_path, index=False)
    interactions_df.to_csv(interactions_output_path, index=False)


if __name__ == "__main__":
    preprocess()
