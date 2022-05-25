import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
import scipy

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

def calc_entropy_with_feature(X, label, thres):
    entropy = 0
    label1 = label[X > thres]
    label2 = label[X <= thres]
    entropy += (label1.shape[0]/label.shape[0]) * calc_entropy(label1)
    entropy += (label2.shape[0]/label.shape[0]) * calc_entropy(label2)
    return entropy

def calc_entropy_with_continuous_feature(X, label, thres):
    entropy = 0
    label1 = label[X > thres]
    label2 = label[X <= thres]
    entropy += (label1.shape[0]/label.shape[0]) * np.std(label1)
    entropy += (label2.shape[0]/label.shape[0]) * np.std(label2)
    return entropy

def info_gain_continuous(X, label, thres):
    entropy = np.std(label)
    entropy_with_feature = calc_entropy_with_continuous_feature(X, label, thres)
    return entropy - entropy_with_feature

def calc_entropy_with_categorical_feature(X, label):
    entropy = 0
    for val in np.unique(X):
        label1 = label[X == val]
        entropy += (label1.shape[0]/label.shape[0]) * calc_entropy(label1)
    return entropy

def info_gain(X, label, thres):
    entropy = np.nan_to_num(calc_entropy(label))
    entropy_with_feature = np.nan_to_num(calc_entropy_with_feature(X, label, thres))
    return entropy - entropy_with_feature

def info_gain2(X, label):
    entropy = np.nan_to_num(calc_entropy(label))
    entropy_with_feature = np.nan_to_num(calc_entropy_with_categorical_feature(X, label))
    return entropy - entropy_with_feature

def search_best_thres(X):
    return np.median(X)

def mse(X, label):
    return np.mean((X - label)**2)

def most_common(label):
    counts = pd.value_counts(label)
    indices = counts.index.tolist()
    return indices[np.argmax(counts.values)]

def id3(root, features, indices, label, min_leaf=10, mode="classifier", params={}):
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
        A = indices[0]; ig_max = 0; thres = np.nan
        for i in indices:
            X = features[:,i]
            if np.unique(X).shape[0] > 1:
                if params["type"][i] == "str":
                    ig = info_gain2(X, label)
                else:
                    X = X.astype(float)
                    thres = np.median(X)
                    ig = info_gain(X, label, thres)
                if ig >= ig_max:
                    A = i; ig_max = ig
    else:
        # regression
        if params["label type"] == "mean":
            n.label = np.mean(label)  
        elif params["label type"] == "median":
            n.label = np.median(label)
        A = indices[0]; min_loss = np.inf; thres = np.nan
        for i in indices:
            vals = features[:,i]
            vals_median = np.median(vals)
            vals_range = [
                (np.min(vals), vals_median),
                (vals_median, np.max(vals))
            ]
            loss = 0; label_subs = []; vals_subs = []
            for vi in vals_range:
                idx_sub = np.argwhere(np.logical_and(vals > vi[0], vals <= vi[1]))[:,0] if vi[0] != vi[1] else np.argwhere(vals == vi[0])[:,0]
                label_sub = label[idx_sub]
                vals_sub = vals[idx_sub]
                label_subs.append(label_sub)
                vals_subs.append(vals_sub)
            
            # calculate the loss
            if params["loss"] == "ks":
                for j in range(len(label_subs)):
                    for k in range(j+1, len(label_subs)):
                        l1 = label_subs[j]
                        l2 = label_subs[k]
                        loss += ks_2samp(l1, l2).pvalue if (l1.shape[0] > 0 and l2.shape[0] > 0) else 1
            elif params["loss"] == "in progress":
                thres = search_best_thres(vals)
                loss += (np.std(label[vals >= thres]) + np.std(label[vals >= thres])) / (np.abs(np.mean(label[vals >= thres]) - np.mean(label[vals >= thres])) + 1e-9)
            if loss < min_loss:
                A = i; min_loss = loss
                
    # examine the branches
    n.feature_index = A
    A_vals = features[:,A]
    if params["verbose"]:
        print("node data:", features.shape, "feature index:", root.feature_index, "range:", root.vi)
    if mode == "classifier":
        if params["type"][A] == "str":
            cat_vals = np.unique(A_vals)
            A_vals_range = list(zip(cat_vals, cat_vals))
        else:
            A_vals = A_vals.astype(float)
            thres = np.median(A_vals)
            A_vals_range = [
                (np.min(A_vals), thres),
                (thres, np.max(A_vals))
            ]
    else:
        A_vals_median = np.median(A_vals)
        if A_vals_median != np.min(A_vals) and A_vals_median != np.max(A_vals):
            A_vals_range = [
                (np.min(A_vals), A_vals_median),
                (A_vals_median, np.max(A_vals))
            ]
        else:
            A_vals_range = [
                (np.min(A_vals), np.min(A_vals)),
                (np.max(A_vals), np.max(A_vals))
            ]

    for vi in A_vals_range:
        idx_with_vi = np.argwhere(np.logical_and(A_vals > vi[0], A_vals <= vi[1]))[:,0] if vi[0] != vi[1] else np.argwhere(A_vals == vi[0])[:,0]
        features_with_vi = features[idx_with_vi,:]
        label_with_vi = label[idx_with_vi]
        n1 = node(); n1.vi = vi; n1.label = n.label
        n.branches.append(n1)
        if label_with_vi.shape[0] > min_leaf and idx_with_vi.shape[0] != label.shape[0]:
            id3(n1, features_with_vi, indices, label_with_vi, min_leaf=min_leaf, mode=mode, params=params)
    return n

def hierarchical_clustering(root, features, indices, min_leaf=10, verbose=False):
    # traversing down the hierarchical structure that maximizes the kl-divergence
    hierarchical_clustering2(root, features, indices, min_leaf, verbose=verbose)
    # labeling the clusters
    q = [root]; leaves = []; counter = 0
    while q:
        cur = q[0]
        q = q[1:]
        if len(cur.branches) > 0:
            for branch in cur.branches:
                q.append(branch)
        else:
            cur.label = counter
            counter += 1

def hierarchical_clustering2(root, features, indices, min_leaf=10, verbose=False):
    n = root
    A = indices[0]; dist_metric = 0
    for i in indices:
        X = features[:,i]
        if np.unique(X).shape[0] > 1:
            thres = np.median(X)
            features1 = features[X > thres,:]
            features2 = features[X <= thres,:]
            X1 = features1[:,A]
            X2 = features2[:,A]
            pval = ks_2samp(X1, X2).pvalue if (len(X1) > 0 and len(X2) > 0) else 1
            dist = (1 / pval if pval > 0 else 1000) 
            if dist > dist_metric:
                A = i; dist_metric = dist

    X = features[:,A]
    thres = np.median(features[:,A])
    n.feature_index = A
    range1 = (-np.inf, thres)
    range2 = (thres, np.inf); nodes = [node(), node()]
    if verbose:
        print("node data:", features.shape, "feature index:", root.feature_index, "range:", root.vi)
    if sum(X > thres) >= min_leaf and sum(X > thres) != X.shape[0]:
        n.branches = [nodes[1]]
        nodes[1].vi = range2
        hierarchical_clustering2(nodes[1], features[X > thres,:], indices, min_leaf, verbose=verbose)
    if sum(X <= thres) >= min_leaf and sum(X <= thres) != X.shape[0]:
        n.branches = [nodes[0]] if len(n.branches) == 0 else [nodes[0], nodes[1]]
        nodes[0].vi = range1
        hierarchical_clustering2(nodes[0], features[X <= thres,:], indices, min_leaf, verbose=verbose)

def label_with_tree(root, feature_row, params, verbose=False):
    f = feature_row[root.feature_index]
    ftype = params["type"][root.feature_index]
    if ftype == "float":
        f = float(f)
    for n1 in root.branches:
        vi = n1.vi
        if vi != "undefined":
            if ftype == "float" and (f >= vi[0] and f <= vi[1]):
                return label_with_tree(n1, feature_row, params, verbose=verbose)
            elif ftype == "str" and (f == vi[0]):
                return label_with_tree(n1, feature_row, params, verbose=verbose)
    if verbose:
        print("feature index:", root.feature_index, "label:", root.label)
    return root.label