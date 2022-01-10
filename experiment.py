import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import base.utils as utils
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import json


def main(data_files, files_to_merge, multi_output, scorer, model, model_params, downproject=False,
         downprojection_model=None, downprojection_params={}, dp_with_metadata=False, proba_threshold=0.5,
         predict_popularity=False, random_seed=None, scale_metadata=False, scale_added_data=False, mode="test",
         files_to_merge_predict=[]):
    if random_seed is not None:
        np.random.seed(random_seed)

    data_train = pd.read_csv(data_files["metadata_train"])
    data_train_ids = pd.DataFrame(data_train["ID"])
    data_test = pd.read_csv(data_files["metadata_test"])
    data_test_ids = pd.DataFrame(data_test["ID"])
    test_metadata = data_test.loc[:, data_test.columns.isin(["ID", "artist", "song", "album"])].copy()

    tags_train = pd.read_csv(data_files["tags_train"])
    tags_train["tags"] = tags_train["tags"].apply(lambda tag: tag.strip("[").strip("]").split(",")).apply(
        lambda tags: [tag.strip().strip("\'") for tag in tags])

    multilabel_binarizer = MultiLabelBinarizer()
    y_tags = multilabel_binarizer.fit_transform(tags_train["tags"])
    tags_train["y"] = y_tags.tolist()
    y_pop = data_train["popularity"].to_numpy()

    if scale_metadata:
        cols = ["danceability", "energy", "key", "mode", "valence", "tempo", "duration_ms"]
        data_train[cols] = StandardScaler().fit_transform(data_train[cols])

    # only csv, prepare data accordingly
    data_train_add = data_train_ids.copy()
    for file_name in files_to_merge:
        df = pd.read_csv(data_files[file_name])
        # TODO VGG RAW split
        if file_name == "vgg_agg":
            df = df.iloc[:, :4097]  # max
        data_train_add = data_train_add.merge(df, on="ID")

    if mode == "predict":
        data_test_add = data_test_ids.copy()
        for file_name in files_to_merge_predict:
            df = pd.read_csv(data_files[file_name])
            # TODO VGG RAW split
            if file_name == "vgg_agg":
                df = df.iloc[:, :4097]  # max
            data_test_add = data_test_add.merge(df, on="ID")

    if downproject:
        if dp_with_metadata:
            df = data_train.merge(data_train_add, on="ID")
            X = df.loc[:, ~df.columns.isin(["ID", "artist", "song", "album", "popularity"])]
            if mode == "predict":
                df = data_test.merge(data_test_add, on="ID")
                X_test = df.loc[:, ~df.columns.isin(["ID", "artist", "song", "album", "popularity"])]
        else:
            X = data_train_add.loc[:, ~data_train_add.columns.isin(["ID"])]
            if mode == "predict":
                X_test = data_test_add.loc[:, ~data_test_add.columns.isin(["ID"])]

        if scale_added_data:
            X = StandardScaler().fit_transform(X)
            if mode == "predict":
                X_test = StandardScaler().fit_transform(X_test)

        print(f"downprojecting X train with {downprojection_model}")

        dp_model = utils.get_model_unsupervised(downprojection_model, downprojection_params)
        X = dp_model.fit_transform(X)

        data_train = pd.concat([data_train, pd.DataFrame(X)], axis=1) if not dp_with_metadata else pd.concat(
            [data_train_ids, pd.DataFrame(X)], axis=1)

        if mode == "predict":
            X_test = dp_model.transform(X_test)
            data_test = pd.concat([data_test, pd.DataFrame(X_test)], axis=1) if not dp_with_metadata else pd.concat(
                [data_test_ids, pd.DataFrame(X_test)], axis=1)
    else:
        data_train = data_train.merge(data_train_add, on="ID")
        if mode == "predict":
            data_test = data_test.merge(data_test_add, on="ID")

    # just ignore instead of one-hot, net helping with generalization
    data_train = data_train.loc[:, ~data_train.columns.isin(["ID", "artist", "song", "album", "popularity"])]
    data_test = data_test.loc[:, ~data_test.columns.isin(["ID", "artist", "song", "album"])]

    # TODO
    ####
    X_pred = []
    y_pred = None
    ####

    clf_base = utils.get_model(model, model_params)

    X_train = data_train.to_numpy()
    y_train = y_pop if predict_popularity else y_tags
    if mode == "predict":
        X_test = data_test.to_numpy()

    if not predict_popularity:
        model = utils.get_wrapper(clf_base, multi_output)
    else:
        model = clf_base

    model.set_data(X_train, X_pred, y_train, y_pred, predict_popularity)
    if mode == "predict":
        print(f"starting predicting for {'popularity' if predict_popularity else 'genres'}")
        model.fit(X_train, y_train)

        if predict_popularity is False:
            pred_proba = model.predict_proba(X_test)
            y_pred = pred_proba >= proba_threshold
            for j in range(len(y_pred)):
                if (~y_pred[j]).all():
                    i_max = np.argmax(pred_proba[j])
                    y_pred[j][i_max] = True
            binarized_pred = multilabel_binarizer.inverse_transform(y_pred)
            y_pred = pd.Series(map(list, binarized_pred))
        else:
            y_pred = model.predict(X_test)

        res_mode = 'tags' if predict_popularity is False else 'popularity'
        res = pd.concat([data_test_ids, pd.DataFrame(y_pred, columns=[res_mode])], axis=1)
        filename = f"pred_{res_mode}.csv"
        res.to_csv(f"results/{filename}", index=False, header=["ID", res_mode])

        print(f"prediction for {res_mode} done")
        print("trying to build final prediction files")

        load_mode = 'tags' if predict_popularity else 'popularity'
        try:
            res_alt = pd.read_csv(f"results/pred_{load_mode}.csv")
        except:
            print(f"{res_mode} file could not be loaded")
            return
        submission = res.merge(res_alt, on="ID")
        submission.to_csv("results/MMSR2021_group_C.csv", index=False)

        urls = pd.read_csv(data_files["youtube_urls"])
        test_metadata = test_metadata.merge(urls, on="ID")
        test_metadata = test_metadata.merge(submission, on="ID")
        test_metadata.to_csv("results/full.csv", index=False)

        if predict_popularity:
            # genre predictions get loaded from csv in case of popularity predictions, need same fixes as dev_tags
            test_metadata["tags"] = test_metadata["tags"].apply(lambda tag: tag.strip("[").strip("]").split(",")).apply(
                lambda tags: [tag.strip().strip("\'") for tag in tags])

        test_metadata.to_json("results/full.json", orient="records")
        print("predictions done")
    else:
        # always use 5 fold as in assignment
        # Attention: uses predict_proba and just errors if model does not support it
        print(f"starting cross validation for {'popularity' if predict_popularity else 'genres'}")
        score, filename = model.cross_validate(scoring=scorer, proba_threshold=proba_threshold,
                                               regression=predict_popularity, data_files=files_to_merge,
                                               scale_added_data=scale_added_data, downproject=downproject,
                                               downprojection_model=downprojection_model,
                                               downprojection_params=downprojection_params)
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
