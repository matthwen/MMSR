from base.ModelBase import BaseModelSklearn
from sklearn import discriminant_analysis


class LinearDiscriminantAnalysis(BaseModelSklearn):
    def __init__(self, param_dict={}):
        BaseModelSklearn.__init__(self, discriminant_analysis.LinearDiscriminantAnalysis(**param_dict))
        self.param_dict = param_dict


class QuadraticDiscriminantAnalysis(BaseModelSklearn):
    def __init__(self, param_dict={}):
        BaseModelSklearn.__init__(self, discriminant_analysis.QuadraticDiscriminantAnalysis(**param_dict))
        self.param_dict = param_dict
