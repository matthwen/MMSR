from base.ModelBase import BaseModelSklearn
from sklearn import svm


class SVM(BaseModelSklearn):
    def __init__(self, param_dict={}):
        BaseModelSklearn.__init__(self, svm.SVC(**param_dict))
        self.param_dict = param_dict


class LinearSVM(BaseModelSklearn):
    def __init__(self, param_dict={}):
        BaseModelSklearn.__init__(self, svm.LinearSVC(**param_dict))
        self.param_dict = param_dict


class SVMRegressor(BaseModelSklearn):
    def __init__(self, param_dict={}):
        BaseModelSklearn.__init__(self, svm.SVR(**param_dict))
        self.param_dict = param_dict
