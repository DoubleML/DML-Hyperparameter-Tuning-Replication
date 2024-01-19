from flaml import AutoML
from sklearn.utils.multiclass import unique_labels

class FlamlRegressorDoubleML:
    _estimator_type = 'regressor'

    def __init__(self, time, estimator_list, metric, *args, **kwargs):
        self.auto_ml = AutoML(*args, **kwargs)
        self.time = time
        self.estimator_list = estimator_list
        self.metric = metric

    def set_params(self, **params):
        self.auto_ml.set_params(**params)
        return self

    def get_params(self, deep=True):
        dict = self.auto_ml.get_params(deep)
        dict["time"] = self.time
        dict["estimator_list"] = self.estimator_list
        dict["metric"] = self.metric
        return dict

    def fit(self, X, y):
        self.auto_ml.fit(X, y, task="regression", time_budget=self.time, estimator_list = self.estimator_list, metric = self.metric, verbose=False)
        self.tuned_model = self.auto_ml.model.estimator
        return self

    def predict(self, x):
        preds = self.tuned_model.predict(x)
        return preds
        
class FlamlClassifierDoubleML:
    _estimator_type = 'classifier'

    def __init__(self, time, estimator_list, metric, *args, **kwargs):
        self.auto_ml = AutoML(*args, **kwargs)
        self.time = time
        self.estimator_list = estimator_list
        self.metric = metric

    def set_params(self, **params):
        self.auto_ml.set_params(**params)
        return self

    def get_params(self, deep=True):
        dict = self.auto_ml.get_params(deep)
        dict["time"] = self.time
        dict["estimator_list"] = self.estimator_list
        dict["metric"] = self.metric
        return dict

    def fit(self, X, y):
        self.classes_ = unique_labels(y)
        self.auto_ml.fit(X, y, task="classification", time_budget=self.time, estimator_list = self.estimator_list, metric = self.metric, verbose=False)
        self.tuned_model = self.auto_ml.model.estimator
        return self

    def predict_proba(self, x):
        preds = self.tuned_model.predict_proba(x)
        return preds