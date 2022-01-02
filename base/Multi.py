from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OneVsRestClassifier
from base.ModelBase import BaseModelSklearn

class MultiOutput(BaseModelSklearn):
    def __init__(self, param_dict={}):
        BaseModelSklearn.__init__(self, MultiOutputClassifier(**param_dict))
        self.param_dict = param_dict


class OneVsRest(BaseModelSklearn):
    def __init__(self, param_dict={}):
        BaseModelSklearn.__init__(self, OneVsRestClassifier(**param_dict))
        self.param_dict = param_dict