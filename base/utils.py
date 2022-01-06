import os
import datetime
import itertools
from sklearn.model_selection import train_test_split
import base.DecisionTree as DecisionTree
import base.SVM as SVM
import base.LinearModels as LinearModels
import base.KNN as KNN
import base.NaiveBayes as NaiveBayes
import base.DiscriminantAnalysis as DiscriminantAnalysis
import torch
from torch.nn.utils import prune
import numpy as np
import json
from sklearn.preprocessing import StandardScaler
from base.StratifiedGroupKFold import StratifiedGroupKFold
from base.Multi import OneVsRest
from base.Multi import MultiOutput
import base.Dummy as Dummy
from sklearn.decomposition import PCA, KernelPCA, FactorAnalysis, TruncatedSVD, FastICA

import pandas as pd


def get_wrapper(model, multi_output):
    return MultiOutput({"estimator": model}) if multi_output else OneVsRest(
        {"estimator": model})


def get_result_folder_for_model(model_name):
    return os.path.join(os.path.dirname(__file__), "..", "results", model_name)


def get_features_labels(data, drop_va=True, drop_mm=True):
    y = data.quadrant if "quadrant" in data else None
    # todo id, remove or convert to numeric?
    X = _drop_labels(data, drop_va=drop_va, drop_mm=drop_mm)

    return X, y


def get_features_mode(data, scale_by_key_strength=False):
    y = data.score_mode if not scale_by_key_strength else data.score_mode.apply(
        lambda x: x if x == 1 else -1) * data.score_key_strength
    # todo id, remove or convert to numeric?
    X = _drop_labels(data)

    return X, y


def _drop_labels(data, drop_quadrant=True, drop_id=True, drop_va=True, drop_mm=True):
    X = data.copy()
    if 'quadrant' in data.columns and drop_quadrant:
        X = X.drop(["quadrant"], axis=1)
    if 'mean_A' in data.columns and drop_va:
        X = X.drop(["mean_A"], axis=1)
    if 'mean_V' in data.columns and drop_va:
        X = X.drop(["mean_V"], axis=1)
    if 'id' in data.columns and drop_id:
        X = X.drop(["id"], axis=1)
    if 'score_mode' in data.columns and drop_mm:
        X = X.drop(["score_mode"], axis=1)
    if 'score_key_strength' in data.columns:
        X = X.drop(["score_key_strength"], axis=1)

    X = X.reset_index().drop('index', axis=1)
    return X


def get_train_test_split(data, test_size, random_state):
    X, y = get_features_labels(data)
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def get_timestamp():
    return str(datetime.datetime.now().strftime("%Y%m%d_%H-%M-%S-%f"))


def get_model(model_name: str, model_params: dict = {}):
    switcher = {
        "tree": DecisionTree.DecisionTree,
        "linear_regression": LinearModels.LinearRegression,
        "random_forest": DecisionTree.RandomForest,
        "svm": SVM.SVM,
        "linear_svm": SVM.LinearSVM,
        "knn": KNN.KNN,
        "gaussian_nb": NaiveBayes.GaussianNaiveBayes,
        "logistic_regression": LinearModels.LogisticRegression,
        "linear_discriminant_analysis": DiscriminantAnalysis.LinearDiscriminantAnalysis,
        "quadratic_discriminant_analysis": DiscriminantAnalysis.QuadraticDiscriminantAnalysis,
        "random_forest_regressor": DecisionTree.RandomForestRegressor,
        "svm_regressor": SVM.SVMRegressor,
        "knn_regressor": KNN.KNNRegressor,
        "dummy": Dummy.Dummy
    }
    return switcher.get(model_name)(model_params)


def get_model_unsupervised(model_name: str, model_params: dict = {}):
    switcher = {
        "pca": PCA,
        "ica": FastICA,
        "factor_analysis": FactorAnalysis,
        "svd": TruncatedSVD,
        "kernel_pca": KernelPCA
    }
    return switcher.get(model_name)(**model_params)


def get_parameter_sets(param_dict):
    walks = []
    params = []

    for key in param_dict:
        if isinstance(param_dict[key], list):
            walks.append([(key, val) for val in param_dict[key]])
    # cartesian product of parameters, executes even if walks is empty
    for combination in itertools.product(*walks):
        param = param_dict.copy()
        for c in combination:
            param[c[0]] = c[1]
        params.append(param)
    return params


def train_and_evaluate(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                       device: torch.device, num_epochs: int,
                       loader_train: torch.utils.data.DataLoader,
                       loader_test: torch.utils.data.DataLoader, prune_training=False, prune_amount=0.1,
                       save_model=False) -> None:
    """
    Auxiliary function for training and evaluating a corresponding model.

    :param model: model instance to train and evaluate
    :param optimizer: optimizer to use for model training
    :param device: device to use for model training and evaluation
    :param num_epochs: amount of epochs for model training
    :param loader_train: data loader supplying the training samples
    :param loader_test: data loader supplying the test samples
    :return: None
    """
    best_accuracy = 0
    for epoch in range(num_epochs):
        # Train model instance for one epoch.
        train_network(
            model=model, data_loader=loader_train, device=device, optimizer=optimizer, prune_training=prune_training,
            prune_amount=prune_amount)

        # Evaluate current model instance.
        performance = test_network(
            model=model, data_loader=loader_train, device=device)

        # Print result of current epoch to standard out.
        print(f'Epoch: {str(epoch + 1).zfill(len(str(num_epochs)))} ' +
              f'/ Loss: {performance[0]:.4f} / Accuracy: {performance[1]:.4f}')

    # Evaluate final model on test data set.
    performance = test_network(model=model, data_loader=loader_test, device=device)
    print(f'\nFinal loss: {performance[0]:.4f} / Final accuracy: {performance[1]:.4f}')
    return model


def train_network(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader,
                  optimizer: torch.optim.Optimizer, device: torch.device = r'cpu', prune_training=False,
                  prune_amount=0.1, class_weights=None) -> None:
    """
    Train specified network for one epoch on specified data loader.

    :param model: network to train
    :param data_loader: data loader to be trained on
    :param optimizer: optimizer used to train network
    :param device: device on which to train network
    :return: None
    """
    model.train()
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    for batch_index, (data, target) in enumerate(data_loader):
        data, target = data.float().to(device), target.long().to(device)
        optimizer.zero_grad()
        output = model(data)
        if max(target) == 4:
            target = target - 1
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    if prune_training:
        parameters_to_prune = (
            (model.fc1, "weight"), (model.fc2, "weight"), (model.fc3, "weight"), (model.fc4, "weight"))
        prune.global_unstructured(
            parameters_to_prune, pruning_method=prune.L1Unstructured, amount=prune_amount
        )
        for child in model.children():
            prune.remove(child, "weight")


def test_network(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader,
                 device: torch.device = r'cpu', class_weights=None):
    """
    Test specified network on specified data loader.

    :param model: network to test on
    :param data_loader: data loader to be tested on
    :param device: device on which to test network
    :return: cross-entropy loss as well as accuracy
    """
    model.eval()
    loss = 0.0
    correct = 0
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.float().to(device), target.long().to(device)
            output = model(data)
            if max(target) == 4:
                target = target - 1
            loss += float(criterion(output, target).item())
            pred = output.max(1, keepdim=True)[1]
            correct += int(pred.eq(target.view_as(pred)).sum().item())

    return loss / len(data_loader.dataset), correct / len(data_loader.dataset)


def get_predictions_network(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader,
                            device: torch.device = r'cpu', class_weights=None):
    """
    Test specified network on specified data loader.

    :param model: network to test on
    :param data_loader: data loader to be tested on
    :param device: device on which to test network
    :return: predictions of target, true tragets
    """
    model.eval()
    preds = []
    true = []
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.float().to(device), target.long().to(device)
            output = model(data).cpu()
            pred = output.max(1, keepdim=True)[1]

            for i, p in enumerate(pred):
                preds.append(p.cpu())
                true.append(target[i].cpu())
            # true.append(target.cpu())

    return preds, true


def _print_results(model_name: str = "Name", params: dict = {}, features: list = None, scores=None, score=None,
                   score_std=None, scoring=None, precisions=None, precision=None,
                   precisions_std=None, recalls=None, recall=None, recall_std=None, maes=None, mae=None, mae_std=None):
    # todo: maybe enable derived functions in subclasses to input their own dict
    if isinstance(scores, np.ndarray):
        scores = scores.tolist()
    if isinstance(recalls, np.ndarray):
        recalls = recalls.tolist()
    if isinstance(precisions, np.ndarray):
        precisions = precisions.tolist()
    if isinstance(maes, np.ndarray):
        maes = maes.tolist()
    results = {"params": params, "scoring_method": scoring, "scores": scores, "score": score,
               "score_std": score_std, "precisions": precisions, "precision": precision,
               "precision_std": precisions_std, "recalls": recalls, "recall": recall,
               "recall_std": recall_std, "mean_absolute_errors": maes, "mae": mae,
               "mae_std": mae_std, "features": features}
    json_string = json.dumps(results, indent=4)
    target_folder = get_result_folder_for_model(model_name)
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    filename = os.path.join(target_folder, f"results_{model_name}_{get_timestamp()}.json")
    with open(filename, "w") as f:
        f.write(json_string)
    return filename


# def get_stratified_group_kfold_splits(data:pd.DataFrame,scale=True,n_splits=6):
#     data = _drop_labels(data,drop_id=False,drop_quadrant=False)
#     X = _drop_labels(data).to_numpy()
#     y = data['quadrant'].to_numpy()
#
#     ids = data['id'].str.split('-')
#     pieces = []
#     for i in ids:
#         pieces.append(i[1])
#     #data['piece'] = pd.Series(pieces)
#     print(data.columns)
#
#     #data_piece = data.groupby('piece')
#     pieces = np.asarray(pieces)
#
#     indices = np.zeros([43,43,2,2607],dtype=int) -1
#     for i in range(2,43):
#         cv = StratifiedGroupKFold(n_splits=i,shuffle=False)
#         for j,idxs in enumerate(cv.split(X, y, pieces)):
#
#             for k, ind in enumerate(idxs[0]):
#                 indices[i, j, 0,k] = ind
#
#             for k, ind in enumerate(idxs[1]):
#                 indices[i, j, 1,k] = ind
#     np.save("StratifiedGroupKFold_Splits_2_42.npy",indices)
#
#
#
#     return X,y,pieces,cv
#
#
#     if scale:
#         sc = StandardScaler()
#         X_train = sc.fit_transform(X_train)
#         X_test = sc.fit_transform(X_test)
#     xtrain_tensor = torch.tensor(X_train)
#     xtest_tensor = torch.tensor(X_test)
#
#     ytrain_tensor = torch.tensor(y_train.values)
#     ytest_tensor = torch.tensor(y_test.values)
#     return xtrain_tensor,xtest_tensor,ytrain_tensor,ytest_tensor


def get_sgk(data, num_splits=5):  # StratifiedGroupKFold
    data = _drop_labels(data, drop_id=False, drop_quadrant=False)
    X = _drop_labels(data).to_numpy()
    y = data['quadrant'].to_numpy() - 1
    cv = StratifiedGroupKFold(num_splits)
    return X, y, cv


def score_revenue_matrix(ys, y_pred, **kwargs):
    if len(ys) != len(y_pred):
        assert KeyError("Length of y_true and y_pred must be equal")
    revenue_matrix = np.array([[5, -5, -2, 4], [-5, 10, 2, -5], [-5, 0, 5, 2], [4, -5, 0, 5]])
    revenue = 0
    for i, y in enumerate(ys):
        revenue += revenue_matrix[y - 1, y_pred[i] - 1]
    return revenue


def write_prediction_file(predictions, ids, target_folder=None):
    target_folder = target_folder if target_folder is not None else get_result_folder_for_model("ensemble")
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    filename = os.path.join(target_folder, f"results_{get_timestamp()}.csv")
    with open(filename, "w") as f:
        f.write("id,quadrant\n")
        for i, y in enumerate(predictions):
            f.write(f"{ids[i]},{y}\n")
    return filename


def get_nn_features():
    return ['essentia_loudness', 'essentia_onset_rate',
            'essentia_spectral_centroid_mean', 'essentia_spectral_centroid_stdev',
            'essentia_spectral_complexity_stdev', 'essentia_spectral_rolloff_mean',
            'essentia_spectral_rolloff_stdev', 'essentia_strong_peak_mean',
            'librosa_bpm', 'librosa_chroma_mean_0', 'librosa_chroma_var_0',
            'librosa_chroma_pct_90_0', 'librosa_chroma_mean_1',
            'librosa_chroma_pct_50_1', 'librosa_chroma_pct_90_1',
            'librosa_chroma_mean_3', 'librosa_chroma_pct_50_3',
            'librosa_chroma_var_4', 'librosa_chroma_pct_50_4',
            'librosa_chroma_pct_90_4', 'librosa_chroma_mean_5',
            'librosa_chroma_var_5', 'librosa_chroma_mean_6', 'librosa_chroma_var_6',
            'librosa_chroma_pct_50_6', 'librosa_chroma_var_7',
            'librosa_chroma_pct_10_7', 'librosa_chroma_mean_9',
            'librosa_chroma_var_9', 'librosa_chroma_mean_11',
            'librosa_chroma_var_11', 'librosa_chroma_pct_50_11',
            'librosa_spectral_bandwidth_mean', 'midlevel_features_melody',
            'midlevel_features_articulation', 'midlevel_features_rhythm_complexity',
            'midlevel_features_rhythm_stability', 'midlevel_features_dissonance',
            'midlevel_features_tonal_stability', 'midlevel_features_minorness']
