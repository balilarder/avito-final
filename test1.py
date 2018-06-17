import pandas as pd
import numpy as np
import os
import lightgbm as lgb
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# get submission ids: dic
cols = ['item_id', 'deal_probability']
df = pd.read_csv('sample_submission.csv', names = cols)
dic = pd.Series.from_csv('sample_submission.csv', header=None).to_dict()

dic.pop('item_id')
print(len(dic.keys()))

# TRAIN: get item id from csv or image..
df = pd.read_csv('train.csv')               # ok, all data loaded successfully
df2 = pd.read_csv('test.csv')
# print(df.columns) 
# print(df.head(3)) 
# print(df.tail(3))
# LGBM
# fill nan
# df = df.fillna(df.mean())
# df['image_top_1'] = df['image_top_1'].fillna((df['image_top_1'].mean()), inplace=True)
# print(df)
#'choose features and train(no image)
catagories = range(100)                     # use deal prob as catagories, round to float 2nd
x_features_train = []
y_labels_train = []

x_test = []
y_test = []

# print(df['deal_probability'])
for a, b, c in zip(df['deal_probability'], 
                               df['image_top_1'],
                               df['price']):
    y = (round(a*100))
    y_labels_train.append(y)
    x_features_train.append(np.array([b, c]))

for a, b in zip(df2['image_top_1'],
                   df2['price']):
    x_test.append(np.array([a, b]))

x_features_train = np.array(x_features_train)
y_labels_train = np.array(y_labels_train)

x_test = np.array(x_test)

# data = np.random.rand(500, 10)  # 500 entities, each contains 10 features
# label = np.random.randint(2, size=500)  # binary target
train_data = lgb.Dataset(x_features_train, label=y_labels_train)

params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': {'l2', 'auc'},
    'num_leaves': 80,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}
gbm = lgb.train(params,
                train_data,
                num_boost_round=20,
                valid_sets=train_data,
                early_stopping_rounds=5)

print('Save model...')
# save model to file
gbm.save_model('model.txt')

# predict
y_pred = gbm.predict(x_test, num_iteration=gbm.best_iteration)
y_pred = y_pred/100.0
# print(y_pred)
print(len(y_pred))
print(len(df2['item_id']))

# print(df2['item_id'])

    

### V1 output
with open('save.csv', 'w') as f:
    wf = csv.writer(f)
    wf.writerow(['item_id', 'deal_probability'])
    
    for i, (a, b) in enumerate(zip(df2['item_id'], y_pred)):
        wf.writerow([a, b])
# ridge regression
f, ax = plt.subplots(figsize=[7,10])
lgb.plot_importance(gbm, max_num_features=50, ax=ax)
plt.title("Light GBM Feature Importance")
plt.savefig('feature_import.png')
