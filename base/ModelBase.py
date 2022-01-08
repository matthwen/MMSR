import json
import os

import sklearn.base
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
import numpy as np
import pandas as pd
import base.utils as utils

import torch


class BaseModelSklearn(sklearn.base.BaseEstimator):
    def __init__(self, model):
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.features = None
        self.model = model
        self.model_name = type(self.model).__name__
        self.classes_ = []

    def set_data(self, X_train, X_test, y_train, y_test, regression=False):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train.astype(int) if regression is False else y_train
        if y_test is not None:
            self.y_test = y_test.astype(int) if regression is False else y_test

    def fit(self, X=None, y=None):
        self.model.fit(X if X is not None else self.X_train, y if y is not None else self.y_train)

    def predict(self, X=None):
        return self.model.predict(X if X is not None else self.X_test).astype(int)

    def predict_proba(self, X=None):
        return self.model.predict_proba(X if X is not None else self.X_test)

    def score(self, X=None, y=None):
        return self.model.score(X if X is not None else self.X_test,
                                y if y is not None else self.y_test)  # This gives the accuracy for Classifier Models, and the R² Score for
        # regression Models

    def accuracy(self):
        y_pred = self.predict()
        return metrics.accuracy_score(self.y_test, y_pred)

    def set_features(self, features):
        self.features = features

    def validate(self, print_results=True):
        score = self.score()
        acc = self.accuracy()
        if self.features is None and type(self.X_test) is pd.DataFrame:
            self.features = self.X_train.columns.tolist()
        if print_results:
            filename = self._print_results(score, acc)
        return score, filename

    def cross_validate(self, print_results=True, cv=5, scoring="accuracy", proba_threshold=0.5, regression=False,
                       data_files=[]):
        # For int/None inputs, if the estimator is a classifier and y is either binary or multiclass,
        # StratifiedKFold is used. In all other cases, Fold is used
        # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html#sklearn.model_selection.cross_validate
        # -> scoring on balanced dataset!

        # todo utils function to get scorer or sth like this
        if scoring == "f1":
            scorer = metrics.make_scorer(metrics.f1_score, average='weighted')
        elif scoring == "balanced_accuracy":
            scorer = metrics.make_scorer(metrics.balanced_accuracy_score)
        elif scoring == "mse":
            scorer = metrics.make_scorer(metrics.mean_squared_error)
        # elif scoring == "revenue":
        #     scorer = metrics.make_scorer(utils.score_revenue_matrix)
        else:
            scorer = metrics.make_scorer(metrics.accuracy_score)

        # results = cross_validate(self.model, self.X_train, self.y_train, cv=cv, scoring=scorer, return_train_score=True)
        kf = KFold(shuffle=True)
        scores = []
        precisions = []
        recalls = []
        maes = []
        i = 1
        X = self.X_train
        y = self.y_train

        for train_index, test_index in kf.split(X):
            print(f"cross validation run: {i}")
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            clf = self.model
            clf.fit(X_train, y_train)
            if not regression:
                pred_proba = clf.predict_proba(X_test)
                y_pred = pred_proba >= proba_threshold
                for j in range(len(y_pred)):
                    if (~y_pred[j]).all():
                        i_max = np.argmax(pred_proba[j])
                        y_pred[j][i_max] = True
                        print("no label probability above threshold, using max")

                average = scoring
                score = metrics.f1_score(y_test, y_pred, average=average, zero_division=0)
                precisions.append(metrics.precision_score(y_test, y_pred, average=average, zero_division=0))
                recalls.append(metrics.recall_score(y_test, y_pred, average=average, zero_division=0))
            else:
                y_pred = clf.predict(X_test)
                score = metrics.mean_squared_error(y_test, y_pred, squared=False)
                maes.append(metrics.mean_absolute_error(y_test, y_pred))

            scores.append(score)
            print(score)
            i += 1

        # scores = results["test_score"]
        # train_scores = results["train_score"]

        scores = np.array(scores)
        precisions = np.array(precisions)
        recalls = np.array(recalls)
        maes = np.array(maes)
        if self.features is None and type(self.X_test) is pd.DataFrame:
            self.features = self.X_train.columns.tolist()
        if print_results:
            # dirty lazy hack
            scoring = "f1" if regression is False else "rmse"
            params = self.model.get_params()
            if "estimator__param_dict" in params.keys():
                real_model = params["estimator"].model_name
                params = params["estimator__param_dict"]
                params["estimator"] = real_model
            if not regression:
                filename = utils._print_results(self.model_name, params, self.features, scores,
                                                scores.mean(), scores.std(), scoring, precisions, precisions.mean(),
                                                precisions.std(), recalls, recalls.mean(), recalls.std(),
                                                data_files=data_files)
            else:
                filename = utils._print_results(self.model_name, params, self.features, scores,
                                                scores.mean(), scores.std(), scoring, maes=maes, mae=maes.mean(),
                                                mae_std=maes.std(), data_files=data_files)
            # filename = utils._print_results(self.model_name, params, self.features, scores,
            #                                 scores.mean(),
            #                                 scores.std(), scoring, train_scores, train_scores.mean(),
            #                                 train_scores.std())
        return scores.mean(), filename

    def _print_results(self, scores=None, score=None, score_std=None, scoring=None, train_scores=None, train_score=None,
                       train_score_std=None):
        params = self.model.get_params()
        # todo: maybe enable derived functions in subclasses to input their own dict
        if isinstance(scores, np.ndarray):
            scores = scores.tolist()
        if isinstance(train_scores, np.ndarray):
            train_scores = train_scores.tolist()
        results = {"params": params, "scoring_method": scoring, "scores": scores, "score": score,
                   "score_std": score_std, "train_scores": train_scores, "train_score": train_score,
                   "train_score_std": train_score_std, "features": self.features}
        json_string = json.dumps(results, indent=4)
        target_folder = utils.get_result_folder_for_model(self.model_name)
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)
        filename = os.path.join(target_folder, f"results_{self.model_name}_{utils.get_timestamp()}.json")
        with open(filename, "w") as f:
            f.write(json_string)
        return filename

# class BaseModelPytorch:
#     def __init__(self, model: torch.nn.Module):
#         self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
#         self.features = None
#         self.model = model
#         self.model_name = type(self.model).__name__
#         self.device = torch.device(r'cuda' if torch.cuda.is_available() else r'cpu')
#         self.loader_train = None
#         self.loader_test = None
#         self.optimizer = None
#         self.num_epochs = 10
#
#     def set_epochs(self, num_epochs: int):
#         self.num_epochs = num_epochs
#
#     def set_loader(self, loader_train: torch.utils.data.DataLoader,
#                    loader_test: torch.utils.data.DataLoader):
#
#         self.loader_train = loader_train
#         self.loader_test = loader_test
#
#     def set_optimizer(self, optimizer: torch.optim.Optimizer):
#         self.optimizer = optimizer
#
#     def set_data(self, X_train, X_test, y_train, y_test):
#         self.X_train = X_train
#         self.X_test = X_test
#         self.y_train = y_train.astype(int)
#         self.y_test = y_test.astype(int)
#
#     def train_network(self) -> None:
#         """
#         Train specified network for one epoch on specified data loader.
#
#         :param model: network to train
#         :param data_loader: data loader to be trained on
#         :param optimizer: optimizer used to train network
#         :param device: device on which to train network
#         :return: None
#         """
#         model = self.model
#         model.train()
#         criterion = torch.nn.CrossEntropyLoss()
#         for batch_index, (data, target) in enumerate(self.loader_train):
#             data, target = data.float().to(self.device), target.long().to(device)
#             optimizer.zero_grad()
#             output = model(data)
#             loss = criterion(output, target)
#             loss.backward()
#             optimizer.step()
#
#     def test_network(self):
#         """
#         Test specified network on specified data loader.
#         :return: cross-entropy loss as well as accuracy
#         """
#         self.model.eval()
#         loss = 0.0
#         correct = 0
#         criterion = torch.nn.CrossEntropyLoss()
#         with torch.no_grad():
#             for data, target in data_loader:
#                 data, target = data.float().to(self.device), target.long().to(self.device)
#                 output = self.model(data)
#                 loss += float(criterion(output, target).item())
#                 pred = output.max(1, keepdim=True)[1]
#                 correct += int(pred.eq(target.view_as(pred)).sum().item())
#
#         return loss / len(data_loader.dataset), correct / len(data_loader.dataset)
#
#     def train_and_evaluate(self) -> None:
#         """
#         Auxiliary function for training and evaluating a corresponding model.
#
#         :param model: model instance to train and evaluate
#         :param optimizer: optimizer to use for model training
#         :param device: device to use for model training and evaluation
#         :param num_epochs: amount of epochs for model training
#         :param loader_train: data loader supplying the training samples
#         :param loader_test: data loader supplying the test samples
#         :return: None
#         """
#         for epoch in range(self.num_epochs):
#             # Train model instance for one epoch.
#             self.train_network()
#
#             # Evaluate current model instance.
#             performance = self.test_network()
#
#             # Print result of current epoch to standard out.
#             print(f'Epoch: {str(epoch + 1).zfill(len(str(self.num_epochs)))} ' +
#                   f'/ Loss: {performance[0]:.4f} / Accuracy: {performance[1]:.4f}')
#
#         # Evaluate final model on test data set.
#         performance = self.test_network()
#         print(f'\nFinal loss: {performance[0]:.4f} / Final accuracy: {performance[1]:.4f}')
