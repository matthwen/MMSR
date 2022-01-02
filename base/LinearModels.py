from base.ModelBase import BaseModelSklearn

from sklearn import linear_model


class LinearRegression(BaseModelSklearn):
    def __init__(self, param_dict={}):
        BaseModelSklearn.__init__(self, linear_model.LinearRegression(**param_dict))
        self.param_dict = param_dict


class LogisticRegression(BaseModelSklearn):
    def __init__(self, param_dict={}):
        BaseModelSklearn.__init__(self, linear_model.LogisticRegression(**param_dict))
        self.param_dict = param_dict

class BayesianRidge(BaseModelSklearn):
    def __init__(self, param_dict={}):
        BaseModelSklearn.__init__(self, linear_model.BayesianRidge(**param_dict))
        self.param_dict = param_dict
