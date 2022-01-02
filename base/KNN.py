from base.ModelBase import BaseModelSklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor


class KNN(BaseModelSklearn):
    def __init__(self, param_dict={}):
        BaseModelSklearn.__init__(self, KNeighborsClassifier(**param_dict))
        self.param_dict = param_dict


class KNNRegressor(BaseModelSklearn):
    def __init__(self, param_dict={}):
        BaseModelSklearn.__init__(self, KNeighborsRegressor(**param_dict))
        self.param_dict = param_dict
