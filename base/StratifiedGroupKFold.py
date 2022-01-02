import numpy as np

class StratifiedGroupKFold:
    def __init__(self,n_splits):
        load = "data/StratifiedGroupKFold_Splits_2_42.npy"
        self.load= np.load(load)
        self.n_splits = n_splits

    def split(self,a,b,c):
        for i in range(self.n_splits):
            train_idxs = self.load[self.n_splits][i][0]
            end_idx = np.where(train_idxs == -1)[0][0]
            train_idxs = train_idxs[:end_idx]


            test_idxs = self.load[self.n_splits][i][1]
            end_idx = np.where(test_idxs == -1)[0][0]
            test_idxs = test_idxs[:end_idx]

            yield train_idxs,test_idxs
