# ods and MTS RecSys Course Competition solution

[Link to competition](https://ods.ai/tracks/recsys-course2021/competitions/competition-recsys-21)

## Competition objective and result
The competition was conducted on MTS Kion streaming service dataset with user-item interactions over a 6 months period and both users and items features. The task was to make 10 recommendations for all users in a test period (1 week). The metric in the competition was map@10.

Baseline solution was a simple top weekly items recommendation for all users and provided 0,091 on public leaderboard.

With a two-stage model of implicit recommendations and gradient boostig I was able to achieve 0,115 and a 4th place on public leaderboard. Private leaderboard scores are not revealed yet.

## End-2-end solution
You can use the following script to reproduce my solution:
```
./full_solution.sh
```

#### Requirements

- Python 3
- NumPy
- Pandas
- Scipy
- Sklearn
- Implicit
- Catboost

## Solution description
My solution included a two-stage model. I used item-item CF from implicit library to generate candidates with their scores and Catboost classifier to predict final ranks with classification objective. Recommendations for cold users were made with Popular items.

Implicit model parameters were chosen on sliding time window cross validation. The best scores were achieved by Cosine recommender model, taking only last 20 interactions for each user. 100 candidates with their scores were generated for each user, filtering all items that user had interactions with.

Implicit candidates were calculated for the last 14 days of the interactions. Then catboost model was trained on positive interactions from the candidates list on last 14 days. Random negative sampling was applied.

For final submission implicit candidates and catboost predictions were recalculated on the whole dataset.

## Features of the Catboost model
![Catboost feature importance](https://github.com/blondered/ods_MTS_RecSys_Challenge_solution/blob/94aa9527850b738de36f7faf89c5201b6c104845/pics/feature_importance.png)

The following features were used in the model.

First-level model scores:
- Implicit scores

Items stats:
- Interactions counts: in last 7 days, in all time
- Timestamp of interactions: standard deviation, 95% quantile difference in days with current day, median differece in days with current day
- Trend slope
- Female watchers fraction in interactions, male watchers fraction in interactions
- Young audience fraction in interactions (younger then 35), older audience fraction in interactions (older then 35)

Item content features:
- Age rating
- Release novelty
- 3 values for genres of the item: minimum, maximum and median of all item genres, encoded with label count method.
- 1 value for countries of the item: maximum of all countries of the item, encoded with label count method.
- 1 value for studios of the item: maximum of all studios of the item, encoded with label count method.
- Content type (movie / series)
- "For kids" boolean feature

User stats:
- User interactions counts: in last 14 days, in all time

User features:
- Sex
- Age
- Income
- "Kids flag" boolean feature

![Shap values](https://github.com/blondered/ods_MTS_RecSys_Challenge_solution/blob/63a4ef4968c0bca35ecedacde436e00507c6d6aa/pics/shap_values.png)
