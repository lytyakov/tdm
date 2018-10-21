import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import gc
import time
from itertools import permutations
from tqdm import tqdm
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from hyperopt import fmin, hp, tpe, Trials, space_eval
directory = ''

types = {'ip': np.uint32,
         'app': np.uint16,
         'device': np.uint16,
         'os': np.uint16,
         'channel': np.uint16,
         'is_attributed': np.uint8,
         'click_id': np.uint32}

def read_data(filename):
    usecols = ['ip', 'device', 'os', 'app', 'channel', 'click_time']
    if 'train' in filename:
        usecols += ['is_attributed'] 
    return pd.read_csv(directory + filename + '.csv', 
                       dtype = types, 
                       usecols = usecols,
                       na_filter = False)
def get_day(df, column = 'click_time'):
    return pd.to_datetime(df[column]).dt.day.astype(np.uint8)
def get_hour(df, column = 'click_time'):
    return pd.to_datetime(df[column]).dt.hour.astype(np.uint8)
def get_minute(df, column = 'click_time'):
    return pd.to_datetime(df[column]).dt.minute.astype(np.uint8)
def get_second(df, column = 'click_time'):   
    return pd.to_datetime(df[column]).dt.second.astype(np.uint8)
def replace_click_time(df):
    df['day'] = get_day(df)
    df['hour'] = get_hour(df)
    df['minute'] = get_minute(df)
    df['second'] = get_second(df)
    return df.drop('click_time', axis = 1)
def mark_test_hours(df):
    return df.eval('test_hour = 1 * ((hour in [4,5,9,10,13,14]) or ((hour in [6,11,15]) & (minute == 0) & (second == 0)))')
def time_to_int(df):
    return df.eval('(day - 6) * (24 * 3600) + hour * 3600 + minute * 60 + second')
def get_columns_combination(df, columns):
    eval_expr = columns[-1]
    power = 0
    for n in range(2, len(columns) + 1):
        power += (len(str(df[columns[1 - n]].max())))
        eval_expr += ' + ' * bool(n) + columns[len(columns) - n] + ' * ' + str(10 ** power)
    return df.eval(eval_expr)
def write_log(text):
    with open(directory + 'ml_log.txt', 'a') as f:
        f.write(time.ctime() + ': ' + text + '\n')
    print(time.ctime() + ': ' + text + '\n')

write_log('Start...')

data = read_data('train')
test = read_data('test')

write_log('Data is loaded. Preprocessing...')

write_log('    Replacing click time, marking test hours...')
data = replace_click_time(data) 
test = replace_click_time(test) 
data = data.query('hour < 17')
data = mark_test_hours(data)
data['test_hour'] = data['test_hour'].astype(np.uint8)
test = mark_test_hours(test)
test['test_hour'] = test['test_hour'].astype(np.uint8)

write_log('    Making train-valid sets...')
train, valid = [], []
certain_day = 9
for df in (train, valid):
    df.append(data.query('day == @certain_day'))
    certain_day -= 1
for i in tqdm(range(len(valid))):
    valid[i] = valid[i].query('(ip < 126413) & (test_hour == 1)')
train_valid_idx = [(0,0)]
totalset = []
train_shape, valid_shape = [], []
for k in tqdm(train_valid_idx):
    totalset.append(pd.concat((train[k[0]], valid[k[1]])).reset_index(drop = True))
    train_shape.append(train[k[0]].shape[0])
    valid_shape.append(valid[k[1]].shape[0])
for i in tqdm(range(len(train))):
    totalset.append(pd.concat((train[i], test)))
    train_shape.append(train[i].shape[0])
gc.collect()

del data, test, train, valid
gc.collect()

write_log('    Making columns combinations...')
for df in tqdm(totalset):
    df['time'] = time_to_int(df)
    df['five'] = (df.time.values // 300).astype(np.uint16)
    df['quarter'] = (df.time.values // 900).astype(np.uint16)
    df['sixty'] = (df.time.values // 3600).astype(np.uint8)
qty_combinations = {'user': ['ip', 'device', 'os'],
                    'user5': ['ip', 'device', 'os', 'five'],   
                    'user15': ['ip', 'device', 'os', 'quarter'],
                    'user60': ['ip', 'device', 'os', 'sixty'],
                    'ip5': ['ip', 'five'],    
                    'ip15': ['ip', 'quarter'],
                    'ip60': ['ip', 'sixty'],      
                    'app5': ['app', 'five'],    
                    'app15':  ['app', 'quarter'],
                    'app60': ['app', 'sixty'], 
                    'chan5': ['channel', 'five'],    
                    'chan15': ['channel', 'quarter'],
                    'chan60': ['channel', 'sixty'], 
                    'app_chan':  ['app', 'channel'],
                    'app_chan5': ['app', 'channel', 'five'],  
                    'app_chan15': ['app', 'channel', 'quarter'],
                    'app_chan60': ['app', 'channel', 'sixty'],                      
                    'user_app': ['ip', 'device', 'os', 'app'],
                    'user_app5': ['ip', 'device', 'os', 'app', 'five'],  
                    'user_app15': ['ip', 'device', 'os', 'app', 'quarter'],
                    'user_app60': ['ip', 'device', 'os', 'app', 'sixty'],
                    'user_chan': ['ip', 'device', 'os', 'channel'],
                    'user_chan5': ['ip', 'device', 'os', 'channel', 'five'], 
                    'user_chan15': ['ip', 'device', 'os', 'channel', 'quarter'],
                    'user_chan60': ['ip', 'device', 'os', 'channel', 'sixty']}

for df in totalset:
    for k in tqdm(qty_combinations):
        df[k] = get_columns_combination(df, qty_combinations[k])

write_log('    Making count columns...')
count_clicks_columns = ['ip', 'ip5', 'ip15', 'ip60',
                        'user', 'user5', 'user15', 'user60', 
                        'app', 'app5', 'app15', 'app60',
                        'channel', 'chan5', 'chan15', 'chan60',
                        'app_chan', 'app_chan5', 'app_chan15', 'app_chan60',
                        'user_app','user_app5','user_app15','user_app60',
                        'user_chan','user_chan5','user_chan15','user_chan60']

for df in totalset:
    for column in tqdm(count_clicks_columns):
        df[column + '_qty'] = df.merge(df[column].value_counts().to_frame('qty'),
                                       left_on = column,
                                       right_index = True,
                                       how = 'left')['qty'].astype(np.uint32)

write_log('    Making share columns...')
for df in totalset:
    for column in ['user_app', 'user_chan']:
        for suf in tqdm(['', '5', '15', '60']):
            df[column + suf + '_share'] = df[column + suf + '_qty'] / df['user' + suf + '_qty']

write_log('    Making time columns...')
for df in tqdm(totalset): 
    df['next_time'] = df.groupby('user')['time'].shift(-1)
    df['time_diff'] = df.eval('next_time - time')
for df in tqdm(totalset):   
    df['hour_std'] = df.merge(df.groupby('user')['hour'].std().to_frame('hour_std'), 
                             left_on = 'user', 
                             right_index = True,
                             how = 'left')['hour_std'].values
    df['min_std'] = df.merge(df.groupby('user')['minute'].std().to_frame('min_std'), 
                             left_on = 'user', 
                             right_index = True,
                             how = 'left')['min_std'].values
    df['sec_std'] = df.merge(df.groupby('user')['second'].std().to_frame('sec_std'), 
                             left_on = 'user', 
                             right_index = True,
                             how = 'left')['sec_std'].values
    df['diff_std'] = df.merge(df.groupby('user')['time_diff'].std().to_frame('diff_std'), 
                             left_on = 'user', 
                             right_index = True,
                             how = 'left')['diff_std'].values

write_log('    Drop columns...')
for df in totalset:
    df.drop(['day', 'time', 'five', 'quarter', 'sixty', 'next_time',
             'user', 'user5', 'user15', 'user60', 
             'ip', 'ip5', 'ip15', 'ip60', 
             'app5', 'app15', 'app60', 
             'chan5', 'chan15', 'chan60', 
             'app_chan', 'app_chan5', 'app_chan15', 'app_chan60', 
             'user_app', 'user_app5', 'user_app15', 'user_app60', 
             'user_chan', 'user_chan5', 'user_chan15', 'user_chan60'], axis = 1, inplace = True)

gc.collect()

write_log('    Make predictors lists...')
for i in range(len(totalset) - 1):
    for j in range(len(totalset[i].columns)):
        assert sorted(totalset[i].columns)[j] == sorted(totalset[i + 1].columns)[j]
                   
predictors = sorted(totalset[0].columns)
predictors.remove('is_attributed')
cat_features = ['app', 'channel', 'device', 'hour',
                'minute', 'os', 'second', 'test_hour']

write_log('Datasets are ready...')

print(totalset[0].head())
print()
print(totalset[1].head())

write_log('Make first validation...')
def get_lgb_score():
    scores = []
    for n, df in enumerate(totalset[:len(train_valid_idx)]):
        print(train_valid_idx[n])
        model = lgb.LGBMClassifier(n_jobs = -1)
        model.fit(X = df.iloc[:train_shape[n],:][predictors],
                  y = df.iloc[:train_shape[n],:]['is_attributed'].values.ravel(),
                  categorical_feature = cat_features,
                  eval_metric = 'auc')
        scores.append(roc_auc_score(df.iloc[train_shape[n]:,:]['is_attributed'].values.ravel(),
                                    model.predict_proba(df.iloc[train_shape[n]:,:][predictors])[:, 1]))
        print('#{:3d} score: {:.5f}'.format(n, np.min(scores)))       
get_lgb_score()

write_log('Make first predictions...')
for n, df in enumerate(totalset[len(train_valid_idx):]):
    
    model = lgb.LGBMClassifier(objective = 'binary',
                               silent = False,
                               n_jobs = -1)
    
    model.fit(X = df[df.is_attributed.notnull()][predictors],
              y = df[df.is_attributed.notnull()]['is_attributed'].values.ravel(),
              categorical_feature=cat_features,
              eval_metric = 'auc')    
    
    predicted = model.predict_proba(df[df.is_attributed.isnull()][predictors])[:, 1]
    submission = pd.DataFrame()
    submission['click_id'] = pd.read_csv(directory + 'test.csv', 
                                         dtype = types, 
                                         usecols = ['click_id'],
                                         na_filter = False)['click_id']
    submission['is_attributed'] = predicted
    path_to_submission = directory + 'lgb_sub0' + str(n + 7) + '.csv'
    submission.to_csv(path_to_submission, index = False)
    
del submission
gc.collect()

write_log('Selecting features...')
max_hyperopt_iters = 5
featur_space = {'col0': hp.choice('col0', ('app', 'None')),
                'col1': hp.choice('col1', ('app15_qty', 'None')),
                'col2': hp.choice('col2',('app5_qty', 'None')),
                'col3': hp.choice('col3',('app60_qty', 'None')),
                'col4': hp.choice('col4', ('app_chan15_qty', 'None')), 
                'col5': hp.choice('col5',('app_chan5_qty', 'None')),
                'col6': hp.choice('col6',('app_chan60_qty', 'None')),
                'col7': hp.choice('col7',('app_chan_qty', 'None')),
                'col8': hp.choice('col8',('app_qty', 'None')),
                'col9': hp.choice('col9',('chan15_qty', 'None')),                
                'col10': hp.choice('col10',('chan5_qty', 'None')),
                'col11': hp.choice('col11',('chan60_qty', 'None')),
                'col12': hp.choice('col12',('channel', 'None')),
                'col13': hp.choice('col13',('channel_qty', 'None')),
                'col14': hp.choice('col14',('device', 'None')),
                'col15': hp.choice('col15',('hour', 'None')),
                'col16': hp.choice('col16',('hour_std', 'None')),
                'col17': hp.choice('col17',('ip15_qty', 'None')),
                'col18': hp.choice('col18',('ip5_qty', 'None')),
                'col19': hp.choice('col19',('ip60_qty', 'None')),
                'col20': hp.choice('col20',('ip_qty', 'None')),
                'col21': hp.choice('col21',('min_std', 'None')),
                'col22': hp.choice('col22',('minute', 'None')),
                'col23': hp.choice('col23',('os', 'None')),
                'col24': hp.choice('col24',('sec_std', 'None')),
                'col25': hp.choice('col25',('second', 'None')),
                'col26': hp.choice('col26',('test_hour', 'None')),
                'col27': hp.choice('col27',('time_diff', 'None')),
                'col28': hp.choice('col28',('user15_qty', 'None')),
                'col29': hp.choice('col29',('user5_qty', 'None')),
                'col30': hp.choice('col30',('user60_qty', 'None')),
                'col31': hp.choice('col31',('user_app15_qty', 'None')),
                'col32': hp.choice('col32',('user_app15_share', 'None')),
                'col33': hp.choice('col33',('user_app5_qty', 'None')),
                'col34': hp.choice('col34',('user_app5_share', 'None')),
                'col35': hp.choice('col35',('user_app60_qty', 'None')),
                'col36': hp.choice('col36',('user_app60_share', 'None')),
                'col37': hp.choice('col37',('user_app_qty', 'None')),
                'col38': hp.choice('col38',('user_app_share', 'None')),
                'col39': hp.choice('col39',('user_chan15_qty', 'None')),
                'col40': hp.choice('col40',('user_chan15_share', 'None')),
                'col41': hp.choice('col41',('user_chan5_qty', 'None')),
                'col42': hp.choice('col42',('user_chan5_share', 'None')),
                'col43': hp.choice('col43',('user_chan60_qty', 'None')),
                'col44': hp.choice('col44',('user_chan60_share', 'None')),
                'col45': hp.choice('col45',('user_chan_qty', 'None')),
                'col46': hp.choice('col46',('user_chan_share', 'None')),
                'col47': hp.choice('col47',('user_qty', 'None')),
                'col48': hp.choice('col48',('diff_std', 'None'))}
selection_result = []
def select_features(features):
    global HYPEROPT_CNT
    scores = []
    for n, df in enumerate(totalset[:len(train_valid_idx)]):
        current_predictors = []
        for k in features.keys():
            if features[k] != 'None':
                current_predictors.append(features[k])
        current_cat_features = [c for c in cat_features[i] if c in current_predictors]
        model = lgb.LGBMClassifier(silent = False,
                                   n_jobs = -1)
        model.fit(X = df.iloc[:train_shape[n],:][current_predictors],
                  y = df.iloc[:train_shape[n],:]['is_attributed'].values.ravel(),
                  categorical_feature = current_cat_features,
                  eval_metric = 'auc')
        scores.append(roc_auc_score(df.iloc[train_shape[n]:,:]['is_attributed'].values.ravel(),
                                   model.predict_proba(df.iloc[train_shape[n]:,:][current_predictors])[:, 1]))
    print('#{:3d} score: {:.5f}, num of features: {}'.format(HYPEROPT_CNT, np.min(scores), len(current_predictors)))

    HYPEROPT_CNT += 1
    selection_result.append(np.min(scores))
    return 1 - np.min(scores) #hyperopt minimizes
HYPEROPT_CNT = 0
trials = Trials()
best_features = fmin(select_features,
                     space = featur_space,
                     algo = tpe.suggest,
                     max_evals = max_hyperopt_iters,
                     trials = trials)

predictors_to_use = [c for c in space_eval(featur_space, best_features).values() if 'None' not in c]
cat_features_to_use = [c for c in cat_features if c in predictors_to_use]

write_log('    Selecting features...')
print('    ' + str(max(selection_result)))
print(predictors_to_use)
print(cat_features_to_use)

write_log('Second predictions...')

for n, df in enumerate(totalset[len(train_valid_idx):]):
    
    model = lgb.LGBMClassifier(objective = 'binary',
                               silent = False,
                               n_jobs = -1)
    
    model.fit(X = df[df.is_attributed.notnull()][predictors_to_use],
              y = df[df.is_attributed.notnull()]['is_attributed'].values.ravel(),
              categorical_feature=cat_features_to_use,
              eval_metric = 'auc')    
    
    predicted = model.predict_proba(df[df.is_attributed.isnull()][predictors_to_use])[:, 1]
    submission = pd.DataFrame()
    submission['click_id'] = pd.read_csv(directory + 'test.csv', 
                                         dtype = types, 
                                         usecols = ['click_id'],
                                         na_filter = False)['click_id']
    submission['is_attributed'] = predicted
    path_to_submission = directory + 'lgb_sub1' + str(n + 7) + '.csv'
    submission.to_csv(path_to_submission, index = False)
    
del submission
gc.collect()

write_log('Tuning...')
params = {'num_leaves': hp.randint('num_leaves', 1000),
          'max_depth': hp.randint('max_depth', 1000),
          'learning_rate': hp.uniform('learning_rate', 1e-15, 0.1),
          'n_estimators': hp.randint('n_estimators', 5000),
          'subsample_for_bin': hp.randint('subsample_for_bin', 5000000), 
          'class_weight': hp.choice('class_weight', ('balanced', None, {0: 1, 1: 333})),
          'min_child_weight': hp.uniform('min_child_weight', 1e-15, 1),
          'min_child_samples': hp.randint('min_child_samples', 1000),
          'subsample': hp.uniform('subsample', 1e-15, 1),
          'subsample_freq': hp.randint('subsample_freq', 1000),
          'colsample_bytree': hp.uniform('colsample_bytree', 1/len(predictors_to_use), 1),
          'reg_alpha': hp.uniform('reg_alpha', 0, 10),
          'reg_lambda': hp.uniform('reg_lambda', 0, 10),
          'random_state': hp.randint('random_state', 100)}
tuning_result = []
def optymize(params):
    global HYPEROPT_CNT
    scores = []
    for n, df in enumerate(totalset[:len(train_valid_idx)]):
        model = lgb.LGBMClassifier(num_leaves=params['num_leaves'],
                                   max_depth=params['max_depth'],
                                   learning_rate=params['learning_rate'],
                                   n_estimators=params['num_leaves'],
                                   subsample_for_bin=params['subsample_for_bin'],
                                   class_weight=params['class_weight'],
                                   min_child_weight=params['min_child_weight'],
                                   min_child_samples=params['min_child_samples'],
                                   subsample=params['subsample'],
                                   subsample_freq=params['subsample_freq'],
                                   colsample_bytree=params['colsample_bytree'],
                                   reg_alpha=params['reg_alpha'],
                                   reg_lambda=params['reg_lambda'],
                                   random_state=params['random_state'],
                                   objective = 'binary',
                                   silent = False,
                                   n_jobs = -1)
        model.fit(X = df.iloc[:train_shape[n],:][predictors_to_use],
                  y = df.iloc[:train_shape[n],:]['is_attributed'].values.ravel(),
                  categorical_feature=cat_features_to_use,
                  eval_metric = 'auc')
        scores.append(roc_auc_score(df.iloc[train_shape[n]:,:]['is_attributed'].values.ravel(),
                                    model.predict_proba(df.iloc[train_shape[n]:, :][predictors_to_use])[:, 1]))
    print('#{:3d} score: {:.5f}'.format(HYPEROPT_CNT, np.min(scores)))

    HYPEROPT_CNT += 1
    tuning_result.append(np.min(scores))
    return 1 - np.min(scores) #hyperopt minimizes

HYPEROPT_CNT = 0
trials = Trials()
best_params = fmin(optymize,
                   space = params,
                   algo = tpe.suggest,
                   max_evals = max_hyperopt_iters,
                   trials = trials)

params_to_use = space_eval(params, best_params)

print(max(tuning_result))

params_to_use = space_eval(params, best_params)
print(params_to_use)

write_log('Third predictions...')
for n, df in enumerate(totalset[len(train_valid_idx):]):
    
    model = lgb.LGBMClassifier(num_leaves=params_to_use['num_leaves'],
                               max_depth=params_to_use['max_depth'],
                               learning_rate=params_to_use['learning_rate'],
                               n_estimators=params_to_use['num_leaves'],
                               subsample_for_bin=params_to_use['subsample_for_bin'],
                               class_weight=params_to_use['class_weight'],
                               min_child_weight=params_to_use['min_child_weight'],
                               min_child_samples=params_to_use['min_child_samples'],
                               subsample=params_to_use['subsample'],
                               subsample_freq=params_to_use['subsample_freq'],
                               colsample_bytree=params_to_use['colsample_bytree'],
                               reg_alpha=params_to_use['reg_alpha'],
                               reg_lambda=params_to_use['reg_lambda'],
                               random_state=params_to_use['random_state'],
                               objective = 'binary',
                               silent = False,
                               n_jobs = -1)
    
    model.fit(X = df[df.is_attributed.notnull()][predictors_to_use],
              y = df[df.is_attributed.notnull()]['is_attributed'].values.ravel(),
              categorical_feature=cat_features_to_use,
              eval_metric = 'auc')    
    
    predicted = model.predict_proba(df[df.is_attributed.isnull()][predictors_to_use])[:, 1]
    submission = pd.DataFrame()
    submission['click_id'] = pd.read_csv(directory + 'test.csv', 
                                         dtype = types, 
                                         usecols = ['click_id'],
                                         na_filter = False)['click_id']
    submission['is_attributed'] = predicted
    path_to_submission = directory + 'lgb_sub2' + str(n + 7) + '.csv'
    submission.to_csv(path_to_submission, index = False)
    
del submission
gc.collect()

