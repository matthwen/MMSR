from sklearn.naive_bayes import GaussianNB
from base.ModelBase import BaseModelSklearn


class GaussianNaiveBayes(BaseModelSklearn):
    def __init__(self, param_dict={}):
        BaseModelSklearn.__init__(self, GaussianNB(**param_dict))
        self.param_dict = param_dict
