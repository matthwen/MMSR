from base.ModelBase import BaseModelSklearn
import base.utils as utils
import os

from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
import sklearn.ensemble
import matplotlib.pyplot as plt


class DecisionTree(BaseModelSklearn):
    def __init__(self, param_dict={}):
        BaseModelSklearn.__init__(self, tree.DecisionTreeClassifier(**param_dict))
        self.param_dict = param_dict

    def print(self, feature_names):
        print(tree.export_text(self.model, feature_names=feature_names))

    def plot(self, feature_names):
        fig = plt.figure(figsize=(25, 20))
        _ = tree.plot_tree(self.model, filled=True, feature_names=feature_names)
        target_folder = utils.get_result_folder_for_model(self.model_name)
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)
        fig.savefig(os.path.join(target_folder, f"decision_tree_{utils.get_timestamp()}.png"))


class RandomForest(BaseModelSklearn):
    def __init__(self, param_dict={}):
        BaseModelSklearn.__init__(self, RandomForestClassifier(**param_dict))
        self.param_dict = param_dict

    def check_n_trees(self, X_train, X_test, y_train, y_test, n_estimators=50, plot=True):
        accs = list()
        scores = list()
        for i in range(1, n_estimators + 1):
            rf = RandomForest({f"n_estimators": i, "random_state": 42})
            rf.fit(X_train, y_train)
            rf.validate(X_test, y_test)
            score = rf.score(X_test, y_test)
            scores.append(score)
            acc = rf.accuracy(X_test, y_test)
            accs.append(acc)
        if plot:
            fig = plt.figure()
            ax = plt.axes()
            ax.plot(accs, label="ACC")
            ax.plot(scores, label="Score")
            ax.legend()
            plt.show()


class RandomForestRegressor(BaseModelSklearn):
    def __init__(self, param_dict={}):
        BaseModelSklearn.__init__(self, sklearn.ensemble.RandomForestRegressor(**param_dict))
