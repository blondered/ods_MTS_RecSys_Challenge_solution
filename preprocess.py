import datetime
import numpy as np
import pandas as pd
                                                                               
# Reading files
users_df = pd.read_csv('data/users.csv')
items_df = pd.read_csv('data/items.csv')
interactions_df = pd.read_csv('data/interactions.csv', 
                              parse_dates = ['last_watch_dt'])
submission = pd.read_csv('data/sample_submission.csv')

# Users info preprocessing
users_df['age'] = users_df['age'].fillna('age_unknown')
users_df['age'] = users_df['age'].astype('category')
users_df['income'] = users_df['income'].fillna('income_unknown')
users_df['income'] = users_df['income'].astype('category')
users_df['sex'] = users_df['sex'].fillna('sex_unknown')
users_df.loc[users_df.sex == 'М', 'sex'] = 'M'
users_df.loc[users_df.sex == 'Ж', 'sex'] = 'F'
users_df['sex'] = users_df['sex'].astype('category')
users_df['kids_flg'] = users_df['kids_flg'].astype('bool')

# Items info preprocessing
items_df['content_type'] = items_df['content_type'].astype('category')
items_df['title'] = items_df['title'].str.lower()
items_df['title_orig'] = items_df['title_orig'].fillna('None') 
items_df.loc[items_df['release_year'] < 1980, 'release_novelty'] = 1
items_df.loc[items_df['release_year'] >= 2020, 'release_novelty'] = 6
novelty = 1
for i in range (1980, 2020, 10):
    novelty += 1
    items_df.loc[(items_df['release_year'] >= i) & 
                 (items_df['release_year'] < i+10), 'release_novelty'] = novelty
items_df = items_df.drop(columns=['release_year'])
items_df['for_kids'] = items_df['for_kids'].fillna(0)
items_df['for_kids'] = items_df['for_kids'].astype('bool')
items_df.loc[items_df.age_rating.isna(), 'age_rating'] = 0
items_df['age_rating'] = items_df['age_rating'].astype('category')
items_df['genres_list'] = items_df['genres'].apply(lambda x:  x.split(', '))
num_genres = pd.Series(np.hstack(items_df['genres_list'].values)).value_counts()
items_df['genres_min'] = items_df['genres_list'].apply(
    lambda x: min([num_genres[el] for el in x]))
items_df['genres_max'] = items_df['genres_list'].apply(
    lambda x: max([num_genres[el] for el in x]))
items_df['genres_med'] = items_df['genres_list'].apply(
    lambda x: (np.median([num_genres[el] for el in x])))
items_df['countries'].fillna('None', inplace = True)
items_df['countries'] = items_df['countries'].str.lower()
items_df['countries_list'] = items_df['countries'].apply(
    lambda x:  x.split(', ') if ', ' in x else [x])
num_countries = pd.Series(np.hstack(items_df['countries_list'].values)).value_counts()
items_df['countries_max'] = items_df['countries_list'] .apply(
    lambda x: max([num_countries[el] for el in x]))
items_df['studios'].fillna('None', inplace = True)
items_df['studios'] = items_df['studios'].str.lower()
items_df['studios_list'] = items_df['studios'].apply(
    lambda x:  x.split(', ') if ', ' in x else [x])
num_countries = pd.Series(np.hstack(items_df['studios_list'].values)).value_counts()
items_df['studios_max'] = items_df['studios_list'] .apply(
    lambda x: max([num_countries[el] for el in x]))
items_df.drop(['countries_list', 'genres_list', 'studios_list'], 
              axis = 1, inplace = True)

# Interactions preprocessing
interactions_df['watched_pct'] = interactions_df['watched_pct'].astype(
    pd.Int8Dtype())
interactions_df['watched_pct'] = interactions_df['watched_pct'].fillna(0)
interactions_df['last_watch_dt'] = pd.to_datetime(
    interactions_df['last_watch_dt'])

# User stats feature ingeneering

def add_user_stats(interactions_df, users_df, split_name = ''):
    '''
    Computes user watches stats for particular interactions date split
    and adds them to users dataframe with specific name
    '''
    user_watch_count_all = interactions_df[
        interactions_df['total_dur'] > 300].groupby(by = 'user_id')['item_id'].count()
    max_date_df = interactions_df['last_watch_dt'].max()
    user_watch_count_last_14 = interactions_boost[
        (interactions_df['total_dur'] > 300) & 
        (interactions_df['last_watch_dt'] >= max_date_df - pd.Timedelta(days = 14))
    ].groupby(by = 'user_id')['item_id'].count()
    user_watch_count_all.name = split_name + "user_watch_cnt_all"
    user_watch_count_last_14.name = split_name + "user_watch_cnt_last_14"
    user_watches = pd.DataFrame(user_watch_count_all).join(user_watch_count_last_14, 
                                                           how = 'outer')
    user_watches.fillna(0, inplace = True)
    cols = user_watches.columns
    user_watches[cols] = user_watches[cols].astype('int64')
    users_df = users_df.join(user_watches, on = 'user_id', how = 'outer')
    users_df[cols] = users_df[cols].fillna(0)
    users_df['age'] = users_df['age'].fillna('age_unknown')
    users_df['income'] = users_df['income'].fillna('income_unknown')
    users_df['sex'] = users_df['sex'].fillna('sex_unknown')
    users_df['kids_flg'] = users_df['kids_flg'].fillna(False)
    return users_df

max_date = interactions_df['last_watch_dt'].max() 
boosting_split_date = max_date - pd.Timedelta(days = 14)  
interactions_boost = interactions_df[
    interactions_df['last_watch_dt'] <= boosting_split_date]
users_df = add_user_stats(interactions_boost, users_df, split_name = 'boost_')
users_df = add_user_stats(interactions_df, users_df, split_name = '')

# Item stats feature ingeneering

def add_item_watches_stats(interactions_df, items_df, item_stats):
    '''
    Computes item watches stats for particular interactions date split
    and adds them to item_stats dataframe
    '''
    def smooth(series, window_size, smoothing_func):
        '''Computes smoothed watches for item'''
        series = np.array(series)
        ext = np.r_[2 *series[0]  - series[window_size-1::-1],
                    series,
                    2 * series[-1] - series[-1:-window_size:-1]]
        weights = smoothing_func(window_size)
        smoothed = np.convolve(weights / weights.sum(), ext, mode='same')
        return smoothed[window_size:-window_size+1]
    
    def trend_slope(series, window_size = 7, smoothing_func = np.hamming):
        '''Computes trend slope for item watches'''
        smoothed = smooth(series, window_size, smoothing_func)
        return smoothed[-1] - smoothed[-2]
    
    keep = item_stats.columns
    max_date = interactions_df['last_watch_dt'].max()
    cols = list(range(7))
    for col in cols:
        watches = interactions_df[
            interactions_df['last_watch_dt'] == 
            max_date - pd.Timedelta(days = 6 - col)]
        item_stats = item_stats.join(
            watches.groupby('item_id')['user_id'].count(), lsuffix = col)
    item_stats.fillna(0, inplace = True)
    new_colnames = ['user_id' + str(i) for i in range(1, 7)] + ['user_id']
    trend_slope_to_row = lambda row: trend_slope(row[new_colnames], 
                                                 window_size = 7)
    item_stats['trend_slope'] = item_stats.apply(trend_slope_to_row, 
                                                 axis = 1)
    item_stats['watched_in_7_days'] = item_stats[new_colnames].apply(
        sum, axis = 1)
    item_stats['watch_ts_quantile_95'] = 0
    item_stats['watch_ts_median'] = 0
    item_stats['watch_ts_std'] = 0
    for item_id in item_stats.index:
        watches = interactions_df[interactions_df['item_id'] == item_id]
        day_of_year = watches['last_watch_dt'].apply(
            lambda x: x.dayofyear).astype(np.int64)
        item_stats.loc[item_id, 'watch_ts_quantile_95'] = \
        day_of_year.quantile(q = 0.95, interpolation = 'nearest')         
        item_stats.loc[item_id, 'watch_ts_median'] = \
        day_of_year.quantile(q = 0.5, interpolation = 'nearest')
        item_stats.loc[item_id, 'watch_ts_std'] =  day_of_year.std()
    item_stats['watch_ts_quantile_95_diff'] = \
    max_date.dayofyear - item_stats['watch_ts_quantile_95'] 
    item_stats['watch_ts_median_diff'] = max_date.dayofyear - \
    item_stats['watch_ts_median'] 
    watched_all_time = interactions_df.groupby('item_id')['user_id'].count()
    watched_all_time.name = 'watched_in_all_time'
    item_stats = item_stats.join(watched_all_time, on = 'item_id', how = 'left')
    item_stats.fillna(0, inplace = True)  
    added_cols = ['trend_slope', 
                  'watched_in_7_days', 
                  'watch_ts_quantile_95_diff', 
                  'watch_ts_median_diff', 
                  'watch_ts_std', 
                  'watched_in_all_time']
    return item_stats[list(keep) + added_cols]
  
def add_age_stats(interactions, item_stats, users_df):
    '''
    Computes watchers age stats for items with particular interactions 
    date split and adds them to item_stats dataframe
    '''
    item_stats.reset_index(inplace = True)
    interactions = interactions.set_index('user_id').join(
        users_df[['user_id', 'sex', 'age', 'income']].set_index('user_id'))
    interactions.reset_index(inplace = True)
    interactions['age_overall'] = interactions['age'].replace(
        to_replace = {'age_18_24': 'less_35', 
                      'age_25_34': 'less_35', 
                      'age_35_44': 'over_35', 
                      'age_45_54': 'over_35', 
                      'age_65_inf': 'over_35', 
                      'age_55_64': 'over_35'},
    inplace = True)
    age_stats = interactions.groupby('item_id')['age_overall'] \
                  .value_counts(normalize = True)
    age_stats = pd.DataFrame(age_stats)
    age_stats.columns = ['value']
    age_stats = age_stats.reset_index().pivot(
        index = 'item_id', columns = 'age_overall', values = 'value').drop(
        'age_unknown', axis = 1)
    age_stats.fillna(0, inplace = True)
    item_stats = item_stats.set_index('item_id').join(age_stats)
    item_stats[['less_35', 'over_35']] = item_stats[['less_35', 'over_35']] \
                  .fillna(0)
    item_stats.rename(columns = {'less_35': 'younger_35_fraction', 
                                 'over_35': 'older_35_fraction'},
                     inplace = True)
    return item_stats

def add_sex_stats(interactions, item_stats, users_df):
    '''
    Computes watchers sex stats for items with particular interactions date split
    and adds them to item_stats dataframe
    '''
    item_stats.reset_index(inplace = True)
    interactions = interactions.set_index('user_id') \
                  .join(users_df[['user_id', 'sex', 'age', 'income']] \
                  .set_index('user_id'))
    interactions.reset_index(inplace = True)
    sex_stats = interactions.groupby('item_id')['sex'] \
                  .value_counts(normalize = True)
    sex_stats = pd.DataFrame(sex_stats)
    sex_stats.columns = ['value']
    sex_stats = sex_stats.reset_index() \
                  .pivot(index = 'item_id', columns = 'sex', values = 'value') \
                  .drop('sex_unknown', axis = 1)
    sex_stats.fillna(0, inplace = True)
    item_stats = item_stats.set_index('item_id').join(sex_stats)
    item_stats[['F', 'M']] = item_stats[['F', 'M']].fillna(0)
    item_stats.rename(columns = {'F': 'female_watchers_fraction', 
                                 'M': 'male_watchers_fraction'})
    return item_stats
                  
# Item stats for submit
item_stats = items_df[['item_id']]
item_stats = item_stats.set_index('item_id')
item_stats = add_item_watches_stats(interactions_df, items_df, item_stats)
item_stats.fillna(0, inplace = True)
item_stats = add_sex_stats(interactions_df, item_stats, users_df)
item_stats = add_age_stats(interactions_df, item_stats, users_df)             
item_stats.to_csv('data/item_stats_for_submit.csv', index = True)
                  
# Item stats for boosting training
item_stats = items_df[['item_id']]
item_stats = item_stats.set_index('item_id')
item_stats = add_item_watches_stats(interactions_boost, items_df, item_stats)
item_stats.fillna(0, inplace = True)
item_stats = add_sex_stats(interactions_boost, item_stats, users_df)
item_stats = add_age_stats(interactions_boost, item_stats, users_df)             
item_stats.to_csv('data/item_stats_for_boost_train.csv', index = True)                 

# Saving preprocessed files
users_df.to_csv('data/users_processed.csv', index=False)
items_df.to_csv('data/items_processed.csv', index=False)
interactions_df.to_csv('data/interactions_processed.csv', index=False)
