# MMSR

#Run the experiment
I just tested it in colab and you can use it by just
!git clone https://github.com/matthwen/MMSR

to download the repo and then
%cd MMSR
!python experiment.py working_config.json

of course data needs to be uploaded in colab and referenced in the config file which i guess needs to be edited outside of colab, maybe i can find a better solution but it seems as it could be done soewhat comfortably in colab as well

of yourse you can run it also locally via commandline if you do the steps above if you have git installed (otherwise just download the files from github), just remove the "!"s and "%" that prefix the lines and it should work
running it via pycharm or whatever python ide you like is of course also an option, just make sure you include "working_config.json" without the " in your unning config for exeriment.py

#Parameter documentation
since comments are not a thing in json here is the doc for the conbfig params:

"data_files": the paths for all the possible files, you can alter the paths to existing files if you have them in a different location or add new files (eg. BOF or word vectors)

"files_to_merge": list of files you want to be merged (can be empty) with the basic metadata. ATTENTION: need to be in csv format (for now) and have an "ID" column

"scorer": dead arameter, just iognore at the moment, it allways uses f1 for genres and mse for popularity

"proba_threshold": probabilistic threshold for when a genre is predicted

"multi_output": if you want to use sklearn multioutput (https://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputClassifier.html) instead of one-versus-all

"predict_popularity": if you want the popularity regression, if false it does genre classification

"random_seed": random seed for reproducability,

"model": the name of the model, following options exist:

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

"model_params": parameter dictionary for your model, just use the parameter names of the sklearn models as keys eg. {"n_estimators":100} for random forests, every parameter not specified in the dict will be sklearn default

"downproject": true for downprojecting the data with eg. pca

"dp_with_metadata": true if added data like BOFs should be downprojected together with the metadata, false only downprojects the added data and leaves the metadata as is

"downprojection_model": name of the model used for downprojection, following options exist:

    "pca": PCA,
    "ica": FastICA,
    "factor_analysis": FactorAnalysis,
    "svd": TruncatedSVD,
    "kernel_pca": KernelPCA

"downprojection_params": parameter dict for the downprojection model, see "model_params"




