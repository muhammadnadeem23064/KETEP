import os.path
from typing import List, Tuple, Union
import pathlib
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgbm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
from itertools import permutations
import shap
import pdpbox
import itertools
from pdpbox import pdp, get_dataset, info_plots


class DeyangAICenter_mlpregressor:
    def __init__(self, weights=''):
        if weights:
            self._model = joblib.load(weights)
            # self._shap_explainer = shap.KernelExplainer(self._model, )
        else:
            self._model = None
            self._shap_explainer = None

        self.features_names = ['temp', 'windspeed', 'humidity', 'holiday', 'month', 'hour']

        self.my_premutations = []
        perm = permutations(self.features_names, 2)
        for i in perm:
            self.my_premutations.append(i)

    def train(self, data_file: str, learning_rate=0.01, seed=42, num_iters=200,
              out_dir='resources/static/generated_images/deayang_ai_mlpregressor'):
        out_dir = pathlib.Path(out_dir)
        out_dir.mkdir(exist_ok=True, parents=True)

        dataset4 = pd.read_csv(data_file, parse_dates=True)
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

            # training model

            # specify parameters via map

            model01 = MLPRegressor(hidden_layer_sizes=(10, 100), activation='relu', solver='adam', random_state=1,
                                   max_iter=5000,
                                   learning_rate='constant', early_stopping=False)

            model01.fit(X_train, y_train)

            y_pred = model01.predict(X_train)
            training_r2_score = r2_score(y_train, y_pred)
            print("train r2 score : ", training_r2_score)
            y_pred = model01.predict(X_val)
            test_r2_score = r2_score(y_val, y_pred)
            print("validation r2 score : ", test_r2_score)

            # select best splits
            if test_r2_score > best_val_score:
                best_val_score = test_r2_score
                self._model = model01
        print('best val score:', best_val_score)
        print(X_val)
        explainer = shap.KernelExplainer(self._model.predict, X_train[:100])
        shap_values = explainer.shap_values(X_train[:100])
        print(shap_values)

        # xai
        print(X_train.select_dtypes(include='float64'))
        self.shap(X_val[:4000], out_dir)

        # self.scatter_shaps(X_train, out_dir)

        #for fname in self.features_names:
            #self.pdp_isolate(X_val[:4000], fname, out_dir)

        #self.pdp_interact(X_val[:4000], ['temp', 'month'], out_dir)

        # self.scatter_shaps(X_val[:5], out_dir)
        print('here')

    def save_model(self, dst: str):
        joblib.dump(self._model, dst)

    def predict_from_file(self, data_file: str, out_dir: pathlib.Path, **kwargs):
        df = pd.read_csv(data_file, parse_dates=True)

        df_to_predict = df.drop(columns=['DateTime'])
        pred = self._model.predict(df_to_predict)
        df['pred'] = pred
        shap_image_files = self.draw_shap_on_test_data(df_to_predict, out_dir)
        df['shap'] = shap_image_files
        return df.to_dict('records')

    # XAI methods
    def feature_importances(self):
        importances = self._model.feature_importance()
        indices = np.argsort(importances)
        plt.title('Feature Importances')
        plt.barh(range(len(indices)), importances[indices], color='g', align='center')
        plt.yticks(range(len(indices)), [self.features_names[i] for i in indices])
        plt.xlabel('Relative Importance')
        plt.savefig('importances.png')
        plt.clf()

    def shap(self, X, out_dir: pathlib.Path):
        explainer = shap.KernelExplainer(self._model.predict, X[:4000])
        shap_values = explainer.shap_values(X[:4000])
        out_dir = pathlib.Path(out_dir)
        shap.summary_plot(shap_values,X[:4000], show=False)
        plt.savefig(out_dir / 'scatter.png')
        plt.clf()

        shap.summary_plot(shap_values, X[:4000], plot_type='bar', show=False)
        plt.savefig(out_dir / 'bar.png')
        plt.clf()
        return {
            # 'waterfall': str(out_dir / 'waterfall.png'),
            'scatter': str(out_dir / 'scatter.png'),
            'bar': str(out_dir / 'bar.png'),
        }

    def draw_shap_on_test_data(self, test_df: pd.DataFrame, out_dir: pathlib.Path):
        # visualize the first prediction's explanation
        explainer = shap.KernelExplainer(self._model.predict, test_df)
        shap_values = explainer.shap_values(test_df)

        shap_image_files = []
        for sampleIdx, _ in test_df.iterrows():
            dest = out_dir / 'waterfall_{}.png'.format(sampleIdx)
            shap.plots._waterfall.waterfall_legacy(explainer.expected_value, shap_values[sampleIdx],feature_names=self.features_names)
            plt.savefig(dest, bbox_inches='tight', pad_inches=0.1)
            plt.clf()
            shap_image_files.append(str(dest))

        return shap_image_files

    def lime(self):
        raise NotImplementedError

    def pdp_isolate(self, X, feature_to_isolate: str, out_dir: pathlib.Path):
        assert feature_to_isolate in self.features_names
        pdp_goals = pdp.pdp_isolate(model=self._model,
                                    dataset=X[:4000],
                                    model_features=self.features_names, feature=feature_to_isolate)
        # plot it
        my_data = []
        my_data.append(pdp.pdp_plot(pdp_goals, feature_to_isolate))
        pdp.pdp_plot(pdp_goals, feature_to_isolate)
        loc = out_dir / 'pdp_{}.png'.format(feature_to_isolate)
        plt.savefig(loc)
        plt.clf()
        return loc

    def pdp_interact(self, X, feature_pair: Union[List, Tuple], out_dir: pathlib.Path):

        for features_to_plot in self.my_premutations:
            inter1 = pdp.pdp_interact(model=self._model, dataset=X[:4000], model_features=self.features_names,
                                      features=features_to_plot)
            pdp.pdp_interact_plot(pdp_interact_out=inter1, feature_names=features_to_plot, plot_type='grid',
                                  plot_pdp=True)  # contour
            loc = out_dir / 'pdp_interact_{}_{}.png'.format(features_to_plot[0], features_to_plot[1])
            plt.savefig(loc)
            plt.clf()

    def scatter_shaps(self, X, out_dir: pathlib.Path):
        out_dir = pathlib.Path(out_dir)
        combinations = itertools.permutations(self.features_names, 2)
        explainer = shap.KernelExplainer(self._model.predict, X[:4000])
        shap_values = explainer.shap_values(X[:4000])
        for feature1, feature2 in combinations:
            shap.dependence_plot(feature1, shap_values, X[:4000], feature_names=self.features_names,
                                 interaction_index=feature2, show=False)
            loc = out_dir / 'pdp_scatter_{}_{}.png'.format(feature1, feature2)
            plt.savefig(loc)
            plt.clf()
