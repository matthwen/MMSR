import os
import glob
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np

import torch
import base.utils as utils


def main(data_files, files_to_merge, multi_output, scorer, model, model_params, downproject=False,
         downprojection_model=None, downprojection_params={}, dp_with_metadata=False, proba_threshold=0.5,
         predict_popularity=False, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)

    data_train = pd.read_csv(data_files["metadata_train"])
    data_train_ids = pd.DataFrame(data_train["ID"])
    data_test = pd.read_csv(data_files["metadata_test"])
    tags_train = pd.read_csv(data_files["tags_train"])
    tags_train["tags"] = tags_train["tags"].apply(lambda tag: tag.strip("[").strip("]").split(",")).apply(
        lambda tags: [tag.strip().strip("\'") for tag in tags])

    multilabel_binarizer = MultiLabelBinarizer()
    y_tags = multilabel_binarizer.fit_transform(tags_train["tags"])
    tags_train["y"] = y_tags.tolist()
    y_pop = data_train["popularity"].to_numpy()

    # TODO support more than csv? how to merge?
    data_train_add = data_train_ids.copy()
    for file_name in files_to_merge:
        df = pd.read_csv(data_files[file_name])
        # TODO VGG RAW split
        if file_name == "vgg_agg":
            df = df.iloc[:, :4097]  # max
        data_train_add = data_train_add.merge(df, on="ID")

    if downproject:
        if dp_with_metadata:
            df = data_train.merge(data_train_add, on="ID")
            X = df.loc[:, ~df.columns.isin(["ID", "artist", "song", "album", "popularity"])]
        else:
            X = data_train_add.loc[:, ~data_train_add.columns.isin(["ID"])]

        print(f"downprojecting X train with {downprojection_model}")
        dp_model = utils.get_model_unsupervised(downprojection_model, downprojection_params)
        X = dp_model.fit_transform(X)

        data_train = pd.concat([data_train, pd.DataFrame(X)], axis=1) if not dp_with_metadata else pd.concat(
            [data_train_ids, pd.DataFrame(X)], axis=1)
        # TODO test/pred data
    else:
        data_train = data_train.merge(data_train_add, on="ID")

    # TODO one hot, maybe?
    data_train = data_train.loc[:, ~data_train.columns.isin(["ID", "artist", "song", "album", "popularity"])]
    data_test = data_test.loc[:, ~data_test.columns.isin(["ID", "artist", "song", "album"])]

    # TODO
    # scorer = sklearn.metrics.log_loss #needs work inside cv function of model base
    X_pred = []
    y_pred = None
    ####

    # model = clf_base = DummyClassifier()
    # model = clf_base = utils.get_model("dummy", {"strategy": "stratified"})
    # model = clf_base = utils.get_model("svm",{"probability":True})
    # model = clf_base = utils.get_model("logistic_regression",{"max_iter":1000})

    clf_base = utils.get_model(model, model_params)

    X_train = data_train.to_numpy()
    y_train = y_pop if predict_popularity else y_tags

    print(f"starting cross validation for {'popularity' if predict_popularity else 'genres'}")
    if not predict_popularity:
        model = utils.get_wrapper(clf_base, multi_output)
    else:
        model = clf_base

    model.set_data(X_train, X_pred, y_train, y_pred, predict_popularity)
    # always use 5 fold as in assignment
    # Attention: uses predict_proba and just errors if model does not support it
    score, filename = model.cross_validate(scoring=scorer, proba_threshold=proba_threshold,
                                           regression=predict_popularity)
    print(f"mean cv score: {score}")
    print(f"result file : {filename} ")


if __name__ == '__main__':
    import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('config_file', help='path to config file', type=str)
args = parser.parse_args()
config_file = args.config_file

with open(config_file, 'r') as fh:
    config = json.load(fh)

main(**config)
