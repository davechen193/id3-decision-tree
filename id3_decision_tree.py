import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

class node:
    def __init__(self, dataval=None):
        self.label = "undefined"
        self.vi = "undefined"
        self.feature_index = -1
        self.branches = []

def calc_entropy(S):
    S_vals, counts = np.unique(S, return_counts=True)
    probs = np.array([count/S.shape[0] for count, s in zip(counts, S_vals)])
    entropy = 0
    for i in range(probs.shape[0]):
        entropy += -probs[i] * np.log(probs[i])/np.log(2)
    return entropy

def calc_entropy_with_feature(X, label):
    entropy = 0
    label1 = label[X > np.median(X)]
    label2 = label[X <= np.median(X)]
    entropy += (label1.shape[0]/label.shape[0]) * calc_entropy(label1)
    entropy += (label2.shape[0]/label.shape[0]) * calc_entropy(label2)
    return entropy

def info_gain(X, label):
    entropy = np.nan_to_num(calc_entropy(label))
    entropy_with_feature = np.nan_to_num(calc_entropy_with_feature(X, label))
    return entropy - entropy_with_feature

def mse(X, label):
    return np.mean((X - label)**2)

def most_common(label):
    counts = pd.value_counts(label)
    indices = counts.index.tolist()
    return indices[np.argmax(counts.values)]

def id3(root, features, indices, label, min_leaf=10, mode="classifier"):
    n = root
    if mode == "classifier":
        # classification
        pos_ratio = sum(label > 0) / label.shape[0] if label.shape[0] > 0 else np.nan
        if pos_ratio == 1:
            n.label = 1
        elif pos_ratio == 0:
            n.label = 0
        else:
            n.label = pos_ratio
        A = indices[0]; ig_max = 0
        for i in indices:
            X = features[:,i]
            ig = info_gain(X, label)
            if ig >= ig_max:
                A = i; ig_max = ig
    else:
        # regression
        n.label = np.mean(label)
        A = indices[0]; min_loss = np.inf
        for i in indices:
            vals = features[:,i]
            vals_median = np.median(vals)
            vals_range = [
                (-np.inf, vals_median),
                (vals_median, np.inf)
            ]
            loss = 0; label_subs = []
            for vi in vals_range:
                idx_sub = np.argwhere(np.logical_and(vals > vi[0], vals <= vi[1]))[:,0]
                label_sub = label[idx_sub]
                label_subs.append(label_sub)
            for j in range(len(label_subs)):
                for k in range(i+1, len(label_subs)):
                    l1 = label_subs[j]
                    l2 = label_subs[k]
                    if j != k:
                        loss += 1 / ks_2samp(l1, l2).pvalue if (len(l1) > 0 and len(l2) > 0) else 2
            if loss < min_loss:
                A = i; min_loss = loss
    # examine the branches
    n.feature_index = A
    A_vals = features[:,A]
    A_vals_median = np.median(A_vals)
    A_vals_range = [
        (-np.inf, A_vals_median),
        (A_vals_median, np.inf)
    ]
    for vi in A_vals_range:
        idx_with_vi = np.argwhere(np.logical_and(features[:,A] > vi[0], features[:,A] <= vi[1]))[:,0]
        features_with_vi = features[idx_with_vi,:]
        label_with_vi = label[idx_with_vi]
        n1 = node(); n1.vi = vi; n1.label = n.label
        n.branches.append(n1)
        if label_with_vi.shape[0] > min_leaf and idx_with_vi.shape[0] != label.shape[0]:
            id3(n1, features_with_vi, indices, label_with_vi, min_leaf=min_leaf, mode=mode)
    return n

def label_with_tree(root, feature_row):
    f = feature_row[root.feature_index]
    for n1 in root.branches:
        vi = n1.vi
        if vi != "undefined":
            if f > vi[0] and f <= vi[1]:
                return label_with_tree(n1, feature_row)
    return root.label