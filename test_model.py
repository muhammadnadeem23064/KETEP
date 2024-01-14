import pathlib
import shutil
import pandas as pd
from pathlib import Path
from itertools import permutations
from resources.libs import model_factory, model_names
from sklearn.model_selection import TimeSeriesSplit
model = model_factory['deayang_ai_mlp_regressor']

#training= deayang_ai_model.DeyangAICenter_randomforest

#training.train('resources/libs/models/deayang_ai_lightbgm/clean_dataset.csv')

model.train('resources/libs/models/deayang_ai_lightbgm/clean_dataset.csv')
# model.save_model('resources/libs/models/deayang_ai_lightbgm/pretrained.pkl')

#model.train('resources/libs/models/deayang_ai_lightbgm/clean_dataset.csv')



# for pdp_interact

"""
my_premutations = []
features_names = ['temp', 'windspeed', 'humidity', 'holiday', 'month', 'hour']
perm = permutations(features_names, 2)
for i in perm:
     my_premutations.append(i)
df = pd.read_csv('resources/libs/models/deayang_ai_lightbgm/clean_dataset.csv', parse_dates=True)
df_to_predict = df.drop(columns=['DateTime',"Amount(total)"])
print(df_to_predict)
out_directory = 'H:\\KETEP-4-03-2023\\out_tmp'

path = pathlib.Path(out_directory)
for j in my_premutations:
     print(j)
     model.scatter_shaps(df, j, path)

"""


#premutation=['temp', 'windspeed', 'humidity', 'holiday', 'month', 'hour']

#my_premutations = []
#features_names = ['temp', 'windspeed', 'humidity', 'holiday', 'month', 'hour']
#perm = permutations(features_names, 2)
#for i in perm:
 #    my_premutations.append(i)


df = pd.read_csv('resources/libs/models/deayang_ai_lightbgm/clean_dataset.csv', parse_dates=True)
df_to_predict = df.drop(columns=['DateTime',"Amount(total)"])
print(df_to_predict)
out_directory = 'H:\\KETEP-4-03-2023\\out_tmp'

path = pathlib.Path(out_directory)

#for j in my_premutations:
    # print(j)
#model.scatter_shaps(df_to_predict, path)

#model.shap("","resources/libs/models/deayang_ai_lightbgm")
#model.scatter_shaps1(df_to_predict,path)



"""
dest = pathlib.Path('out_tmp')
dest.mkdir(exist_ok=True, parents=True)
model.predict_from_file('resources/libs/models/deayang_ai_lightbgm/samples.csv', dest)
shutil.rmtree(dest)
# model.feature_importances()

"""


#model.train('resources/libs/models/deayang_ai_lightbgm/clean_dataset.csv')
#model.save_model('resources/libs/models/deayang_ai_randomforest/pretrained.pkl')



"""
dataset4 = pd.read_csv("resources/libs/models/deayang_ai_lightbgm/clean_dataset.csv", parse_dates=True)
train_set4 = dataset4.set_index('DateTime', drop=True)

# train_set에서 y와 X 분리
y4 = train_set4['Amount(total)']
X4 = train_set4.drop(['Amount(total)'], axis=1)
columns4 = X4.columns
X4 = X4.values
y4 = y4.values

tscv = TimeSeriesSplit()
best_val_score = -1
for train_index, val_index in tscv.split(X4):
    X_train, X_val = X4[train_index], X4[val_index]
    y_train, y_val = y4[train_index], y4[val_index]
    X_train = pd.DataFrame(X_train, columns=columns4)
    X_val = pd.DataFrame(X_val, columns=columns4)
print(X_train)

model.shap(X_train, "resources/libs/models/deayang_ai_lightbgm")
"""
