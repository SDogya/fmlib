import shap
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score


class BaseFeatureSelector(ABC):
    def __init__(self, n_folds=5, random_state=None):
        self.n_folds = n_folds
        self.importances = None
        self.random_state = random_state

    def fit(self, x: np.ndarray, y: np.ndarray, columns: list, **importances_kwargs):
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True,
                              random_state=self.random_state)

        self.importances = pd.DataFrame({'feature': columns})

        for i, (train_index, val_index) in enumerate(skf.split(x, y)):
            print(f'FOLD: {i}')
            x_train, y_train = x[train_index], y[train_index]
            x_val, y_val = x[val_index], y[val_index]

            model = LGBMClassifier(max_depth=5, n_estimators=500,
                                   learning_rate=0.05, verbose=-100,
                                   random_state=self.random_state)
            model.fit(x_train, y_train)

            imp = self._get_importances_from_model(model, x_val, y_val, **importances_kwargs)

            self.importances[f'importance_{i}'] = imp

    def get_selected_features(self, threshold: float):
        assert self.importances is not None, 'Сначала нужно обучить, вызвав метод fit'

        # для 0 итерации отдельно
        imps = self.importances.loc[:, ['feature', 'importance_0']].sort_values('importance_0', ascending=False)
        imps['importance_0'] /= imps['importance_0'].sum()
        imps['cumsum'] = imps['importance_0'].cumsum()
        features = imps.loc[imps['cumsum'] <= threshold, 'feature'].tolist()

        best_features = set(features)
        for i in range(1, self.n_folds):
            imps = self.importances.loc[:, ['feature', f'importance_{i}']].sort_values(f'importance_{i}',
                                                                                       ascending=False)
            imps[f'importance_{i}'] /= imps[f'importance_{i}'].sum()
            imps['cumsum'] = imps[f'importance_{i}'].cumsum()
            features = imps.loc[imps['cumsum'] <= threshold, 'feature'].tolist()

            best_features &= set(features)

        return list(best_features)

    @abstractmethod
    def _get_importances_from_model(self, model, x: np.ndarray, y: np.ndarray, **kwargs):
        pass


class LGMFeatureSelection(BaseFeatureSelector):
    def __init__(self, n_folds=5, random_state=None):
        super().__init__(n_folds, random_state)

    def _get_importances_from_model(self, model, x, y, importance_type='split'):
        return model.booster_.feature_importance(importance_type=importance_type)


class ShapFeatureSelection(BaseFeatureSelector):
    def __init__(self, n_folds=5, random_state=None):
        super().__init__(n_folds, random_state)

    def _get_importances_from_model(self, model, x: np.ndarray, y: np.ndarray, is_multiclass=False,
                                    feature_perturbation='tree_path_dependent'):
        explainer = shap.TreeExplainer(model, feature_perturbation=feature_perturbation)
        shap_values = explainer.shap_values(x)

        if isinstance(shap_values, list):
            if is_multiclass:
                importances = []
                for cls_ in shap_values:
                    c_sum = np.abs(cls_).mean(0)
                    importances.append(c_sum.reshape(1, -1))
                importances = np.concatenate(importances).mean(0)
            else:
                importances = np.abs(shap_values[1]).mean(0)
        else:
            importances = np.abs(shap_values).mean(0)
        return importances
