from base.ModelBase import BaseModelSklearn
from sklearn.dummy import DummyClassifier, DummyRegressor
import sklearn.dummy


class Dummy(BaseModelSklearn):
    def __init__(self, param_dict={}):
        BaseModelSklearn.__init__(self, DummyClassifier(**param_dict))
        self.param_dict = param_dict


class DummyRegressor(BaseModelSklearn):
    def __init__(self, param_dict={}):
        BaseModelSklearn.__init__(self, sklearn.dummy.DummyRegressor(**param_dict))
        self.param_dict = param_dict
