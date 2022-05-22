from typing import List

from numpy import ndarray
from sklearn.metrics import accuracy_score
import numpy as np


def get_npcer(false_negative: int, true_positive: int):
    return false_negative / (false_negative + true_positive + 1e-9)


def get_apcer(false_positive: int, true_negative: int):
    return false_positive / (true_negative + false_positive + 1e-9)


def get_acer(apcer: float, npcer: float):
    return (apcer + npcer) / 2.0


####
# In our work negative should be 0, and positive should be 1.
# We use 1 - acer to get the correct acer in the end.
####
def get_metrics(pred: ndarray, targets: ndarray):
    negative_indices = targets == 1
    positive_indices = targets == 0

    false_positive = (pred[negative_indices] == 1).sum()
    false_negative = (pred[positive_indices] == 0).sum()

    true_positive = (pred[positive_indices] == 1).sum()
    true_negative = (pred[negative_indices] == 0).sum()

    npcer = get_npcer(false_negative, true_positive)
    apcer = get_apcer(false_positive, true_negative)

    acer = get_acer(apcer, npcer)

    return true_negative, false_positive, false_negative, true_positive, acer, apcer, npcer

def get_hter(pred: ndarray, targets: ndarray):
    negative_indices = targets == 1
    positive_indices = targets == 0
    # print('negative_indices: ', negative_indices)
    # print('positive_indices: ', positive_indices)
    # print('pred: ', pred)

    false_positive = (pred[negative_indices] == 1).sum()
    false_negative = (pred[positive_indices] == 0).sum()

    true_positive = (pred[positive_indices] == 1).sum()
    true_negative = (pred[negative_indices] == 0).sum()

    far = false_positive / (true_negative+false_positive+1e-9)
    frr = false_negative / (true_positive+false_negative+1e-9)
    hter = (far + frr) / 2

    return hter, far, frr


def get_threshold(probs: ndarray, grid_density: int = 10):
    min_, max_ = min(probs), max(probs)
    thresholds = [min_]
    for i in range(grid_density + 1):
        thresholds.append(min_ + (i * (max_ - min_)) / float(grid_density))
    # thresholds.append(1.1)
    return thresholds


def eval_from_scores(scores: ndarray, targets: ndarray):
    thrs = get_threshold(scores)

    acc = 0.0
    best_thr = -1
    for thr in thrs:

        acc_new = accuracy_score(targets, scores >= thr)
        if acc_new > acc:
            best_thr = thr
            acc = acc_new
    predict = np.array([1 if score > best_thr else 0 for score in scores])

    return get_metrics(scores >= best_thr, targets), best_thr, acc, predict

def eval_from_scores_thr(scores: ndarray, targets: ndarray, thr):

    acc = accuracy_score(targets, scores >= thr)
    predict = np.array([1 if score > thr else 0 for score in scores])

    return get_metrics(scores >= thr, targets), thr, acc, predict

def eval_from_scores_hter(scores: ndarray, targets: ndarray):
    thrs = get_threshold(scores)

    acc = 0.0
    best_thr = -1
    for thr in thrs:

        acc_new = accuracy_score(targets, scores >= thr)
        if acc_new > acc:
            best_thr = thr
            acc = acc_new
    predict = np.array([1 if score > best_thr else 0 for score in scores])

    return get_hter(scores >= best_thr, targets), best_thr, acc, predict

