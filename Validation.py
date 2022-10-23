# -*- coding: utf-8 -*-
import torch
import numpy as np
from joblib import Parallel, delayed
from itertools import chain


def get_pr(pred, label, prop):
    tp = np.array([0] * 6)
    fp = np.array([0] * 6)
    fn = np.array([0] * 6)

    offset = 0
    if prop == "zero":
        offset = 3
    elif prop == "null":
        pass
    elif prop == "pad":
        return tp, fp, fn

    if pred == label:
        if pred == 0:
            tp[offset] += 1
        elif pred == 1:
            tp[offset + 1] += 1
        elif pred == 2:
            tp[offset + 2] += 1
    else:
        if label == 0:
            fn[offset] += 1
            if pred == 1:
                fp[offset + 1] += 1
            elif pred == 2:
                fp[offset + 2] += 1
        elif label == 1:
            fn[offset + 1] += 1
            if pred == 0:
                fp[offset] += 1
            elif pred == 2:
                fp[offset + 2] += 1
        elif label == 2:
            fn[offset + 2] += 1
            if pred == 0:
                fp[offset] += 1
            elif pred == 1:
                fp[offset + 1] += 1
        elif label == 3:
            if pred == 0:
                fp[offset] += 1
            elif pred == 1:
                fp[offset + 1] += 1
            elif pred == 2:
                fp[offset + 2] += 1
    return tp, fp, fn


def get_prs(batch_pred, batch_label, batch_property):
    batch_pred_flatten = list(chain.from_iterable(batch_pred))
    batch_label_flatten = list(chain.from_iterable(batch_label))
    batch_property_flatten = list(chain.from_iterable(batch_property))

    # paralell = Parallel(n_jobs=-1)([delayed(_get_prs)(pred, label, prop) for pred, label, prop in zip(batch_pred_flatten, batch_label_flatten, batch_property_flatten)])
    # ret = np.array(paralell).sum(axis=0)
    # tp = ret[0]
    # fp = ret[1]
    # fn = ret[2]

    tp = np.array([0] * 6)
    fp = np.array([0] * 6)
    fn = np.array([0] * 6)
    for pred, label, prop in zip(batch_pred_flatten, batch_label_flatten, batch_property_flatten):
        one_tp, one_fp, one_fn = _get_prs(pred, label, prop)
        tp = tp + one_tp
        fp = fp + one_fp
        fn = fn + one_fn
    return tp, fp, fn

def get_prs2(batch_pred, batch_label, batch_property):
    tp = np.array([0] * 6)
    fp = np.array([0] * 6)
    fn = np.array([0] * 6)
    for preds, labels, props in zip(batch_pred, batch_label, batch_property):
        for pred, label, prop in zip(preds, labels, props):
            one_tp, one_fp, one_fn = _get_prs(pred, label, prop)
            tp = tp + one_tp
            fp = fp + one_fp
            fn = fn + one_fn
    return tp, fp, fn

def _get_prs(pred, label, prop):
    tp = np.array([0] * 6)
    fp = np.array([0] * 6)
    fn = np.array([0] * 6)
    if type(pred) is not list:
        one_tp, one_fp, one_fn = get_pr(pred, label, prop)
        tp = tp + one_tp
        fp = fp + one_fp
        fn = fn + one_fn

    elif type(pred) is list:
        item = pred
        for pred in item:
            one_tp, one_fp, one_fn = get_pr(pred, label, prop)
            tp = tp + one_tp
            fp = fp + one_fp
            fn = fn + one_fn
    return tp, fp, fn


def calculate_f(tp, fp, fn):
    # micro F score
    _tp = sum(tp)
    _fp = sum(fp)
    _fn = sum(fn)
    _p = _tp / (_tp + _fp) if (_tp + _fp) > 0 else 0.0
    _r = _tp / (_tp + _fn) if (_tp + _fn) > 0 else 0.0
    _f = (2 * _p * _r) / (_p + _r) if (_p + _r) > 0 else 0.0
    return _f


def score_f(y_pred, y_label, y_property):
    tp, fp, fn = get_prs(y_pred, y_label, y_property)
    all_score = calculate_f(tp, fp, fn)
    dep_score = calculate_f(tp[0:3], fp[0:3], fn[0:3])
    zero_score = calculate_f(tp[3:6], fp[3:6], fn[3:6])

    return all_score, dep_score, zero_score, tp, fp, fn


def combine_labels(ga_labels, ni_labels, wo_labels):
    ret = []
    for ga_label, ni_label, wo_label in zip(ga_labels, ni_labels, wo_labels):
        if ga_label == 1:
            ret.append(0)
            continue
        if ni_label == 1:
            ret.append(2)
            continue
        if wo_label == 1:
            ret.append(1)
            continue
        ret.append(3)
    return ret


def get_pr_numbers(y_pred, y_label, y_property, type='sentence'):
    if type == 'pair':
        y_label = y_label.unsqueeze(1).data.cpu().numpy()
        y_pred = y_pred.unsqueeze(1).data.cpu().numpy()
        y_property = np.expand_dims(y_property, axis=1)
    return get_prs(y_pred, y_label, y_property)

def get_pr_numbers2(y_pred, y_label, y_property):
    return get_prs2(y_pred, y_label, y_property)

def get_f_score(tp, fp, fn):
    all_score = calculate_f(tp, fp, fn)
    dep_score = calculate_f(tp[0:3], fp[0:3], fn[0:3])
    zero_score = calculate_f(tp[3:6], fp[3:6], fn[3:6])
    return all_score, dep_score, zero_score


if __name__ == "__main__":
    import torch

    tp, fp, fn = get_pr_numbers(torch.tensor([[0,1,2,0,1,2,0,1,2]]), torch.tensor([[0,1,2,0,1,2,-1,-1,-1]]), np.array([['dep', 'dep', 'dep', 'zero', 'zero', 'zero', 'pad', 'pad', 'pad']]))
    score_all, score_dep, score_zero = get_f_score(tp, fp, fn)
    assert score_all == 1.0
    assert score_dep == 1.0
    assert score_zero == 1.0
    assert list(tp) == [1, 1, 1, 1, 1, 1]
    assert list(fp) == [0, 0, 0, 0, 0, 0]
    assert list(fn) == [0, 0, 0, 0, 0, 0]

    tp, fp, fn = get_pr_numbers(torch.tensor([[1,2,0,1,2,0,1,2,0]]), torch.tensor([[0,1,2,0,1,2,-1,-1,-1]]), np.array([['dep', 'dep', 'dep', 'zero', 'zero', 'zero', 'pad', 'pad', 'pad']]))
    score_all, score_dep, score_zero = get_f_score(tp, fp, fn)
    assert score_all == 0.0
    assert score_dep == 0.0
    assert score_zero == 0.0
    assert list(tp) == [0, 0, 0, 0, 0, 0]
    assert list(fp) == [1, 1, 1, 1, 1, 1]
    assert list(fn) == [1, 1, 1, 1, 1, 1]

    tp, fp, fn = get_pr_numbers(torch.tensor([[0,1,2,0,1,2,0,1,2]]), torch.tensor([[3,3,3,3,3,3,-1,-1,-1]]), np.array([['dep', 'dep', 'dep', 'zero', 'zero', 'zero', 'pad', 'pad', 'pad']]))
    score_all, score_dep, score_zero = get_f_score(tp, fp, fn)
    assert score_all == 0.0
    assert score_dep == 0.0
    assert score_zero == 0.0
    assert list(tp) == [0, 0, 0, 0, 0, 0]
    assert list(fp) == [1, 1, 1, 1, 1, 1]
    assert list(fn) == [0, 0, 0, 0, 0, 0]

    tp, fp, fn = get_pr_numbers(torch.tensor([[3,3,3,3,3,3,3,3,3]]), torch.tensor([[0,1,2,0,1,2,-1,-1,-1]]), np.array([['dep', 'dep', 'dep', 'zero', 'zero', 'zero', 'pad', 'pad', 'pad']]))
    score_all, score_dep, score_zero = get_f_score(tp, fp, fn)
    assert score_all == 0.0
    assert score_dep == 0.0
    assert score_zero == 0.0
    assert list(tp) == [0, 0, 0, 0, 0, 0]
    assert list(fp) == [0, 0, 0, 0, 0, 0]
    assert list(fn) == [1, 1, 1, 1, 1, 1]

    tp, fp, fn = get_pr_numbers(torch.tensor([[0,0,0,0,0,0,0,0,0]]), torch.tensor([[0,1,2,0,1,2,-1,-1,-1]]), np.array([['dep', 'dep', 'dep', 'zero', 'zero', 'zero', 'pad', 'pad', 'pad']]))
    score_all, score_dep, score_zero = get_f_score(tp, fp, fn)
    assert score_all == 1/3
    assert score_dep == 1/3
    assert score_zero == 1/3
    assert list(tp) == [1, 0, 0, 1, 0, 0]
    assert list(fp) == [2, 0, 0, 2, 0, 0]
    assert list(fn) == [0, 1, 1, 0, 1, 1]

    tp, fp, fn = get_pr_numbers(torch.tensor([[1,1,1,1,1,1,1,1,1]]), torch.tensor([[0,1,2,0,1,2,-1,-1,-1]]), np.array([['dep', 'dep', 'dep', 'zero', 'zero', 'zero', 'pad', 'pad', 'pad']]))
    score_all, score_dep, score_zero = get_f_score(tp, fp, fn)
    assert score_all == 1/3
    assert score_dep == 1/3
    assert score_zero == 1/3
    assert list(tp) == [0, 1, 0, 0, 1, 0]
    assert list(fp) == [0, 2, 0, 0, 2, 0]
    assert list(fn) == [1, 0, 1, 1, 0, 1]

    tp, fp, fn = get_pr_numbers(torch.tensor([[2,2,2,2,2,2,2,2,2]]), torch.tensor([[0,1,2,0,1,2,-1,-1,-1]]), np.array([['dep', 'dep', 'dep', 'zero', 'zero', 'zero', 'pad', 'pad', 'pad']]))
    score_all, score_dep, score_zero = get_f_score(tp, fp, fn)
    assert score_all == 1/3
    assert score_dep == 1/3
    assert score_zero == 1/3
    assert list(tp) == [0, 0, 1, 0, 0, 1]
    assert list(fp) == [0, 0, 2, 0, 0, 2]
    assert list(fn) == [1, 1, 0, 1, 1, 0]

    tp = [3,0,0,1,0,0]
    fp = [32,59,96,43,102,129]
    fn = [21,13,5,5,2,0]
    score_all, score_dep, score_zero = get_f_score(tp, fp, fn)
    assert score_all - 0.01553 < 1e-4
    assert score_dep - 0.02586 < 1e-4
    assert score_zero - 0.00707 < 1e-4

    tp = [3, 0, 0, 2, 0, 0]
    fp = [2, 4, 3, 1, 2, 3]
    fn = [2, 1, 2, 1, 0, 0]
    score_all, score_dep, score_zero = get_f_score(tp, fp, fn)
    assert score_all - 0.322581 < 1e-4
    assert score_dep - 0.30000 < 1e-4
    assert score_zero - 0.363636 < 1e-4

    score_all, score_dep, score_zero, tp, fp, fn = score_f(torch.tensor([[0,1,2,0,1,2,0,1,2]]), torch.tensor([[0,1,2,0,1,2,-1,-1,-1]]), np.array([['dep', 'dep', 'dep', 'zero', 'zero', 'zero', 'pad', 'pad', 'pad']]))
    assert score_all == 1.0
    assert score_dep == 1.0
    assert score_zero == 1.0
    assert list(tp) == [1, 1, 1, 1, 1, 1]
    assert list(fp) == [0, 0, 0, 0, 0, 0]
    assert list(fn) == [0, 0, 0, 0, 0, 0]

    score_all, score_dep, score_zero, tp, fp, fn = score_f(torch.tensor([[1,2,0,1,2,0,1,2,0]]), torch.tensor([[0,1,2,0,1,2,-1,-1,-1]]), np.array([['dep', 'dep', 'dep', 'zero', 'zero', 'zero', 'pad', 'pad', 'pad']]))
    assert score_all == 0.0
    assert score_dep == 0.0
    assert score_zero == 0.0
    assert list(tp) == [0, 0, 0, 0, 0, 0]
    assert list(fp) == [1, 1, 1, 1, 1, 1]
    assert list(fn) == [1, 1, 1, 1, 1, 1]

    score_all, score_dep, score_zero, tp, fp, fn = score_f(torch.tensor([[0,1,2,0,1,2,0,1,2]]), torch.tensor([[3,3,3,3,3,3,-1,-1,-1]]), np.array([['dep', 'dep', 'dep', 'zero', 'zero', 'zero', 'pad', 'pad', 'pad']]))
    assert score_all == 0.0
    assert score_dep == 0.0
    assert score_zero == 0.0
    assert list(tp) == [0, 0, 0, 0, 0, 0]
    assert list(fp) == [1, 1, 1, 1, 1, 1]
    assert list(fn) == [0, 0, 0, 0, 0, 0]

    score_all, score_dep, score_zero, tp, fp, fn = score_f(torch.tensor([[3,3,3,3,3,3,3,3,3]]), torch.tensor([[0,1,2,0,1,2,-1,-1,-1]]), np.array([['dep', 'dep', 'dep', 'zero', 'zero', 'zero', 'pad', 'pad', 'pad']]))
    assert score_all == 0.0
    assert score_dep == 0.0
    assert score_zero == 0.0
    assert list(tp) == [0, 0, 0, 0, 0, 0]
    assert list(fp) == [0, 0, 0, 0, 0, 0]
    assert list(fn) == [1, 1, 1, 1, 1, 1]

    score_all, score_dep, score_zero, tp, fp, fn = score_f(torch.tensor([[0,0,0,0,0,0,0,0,0]]), torch.tensor([[0,1,2,0,1,2,-1,-1,-1]]), np.array([['dep', 'dep', 'dep', 'zero', 'zero', 'zero', 'pad', 'pad', 'pad']]))
    assert score_all == 1/3
    assert score_dep == 1/3
    assert score_zero == 1/3
    assert list(tp) == [1, 0, 0, 1, 0, 0]
    assert list(fp) == [2, 0, 0, 2, 0, 0]
    assert list(fn) == [0, 1, 1, 0, 1, 1]

    score_all, score_dep, score_zero, tp, fp, fn = score_f(torch.tensor([[1,1,1,1,1,1,1,1,1]]), torch.tensor([[0,1,2,0,1,2,-1,-1,-1]]), np.array([['dep', 'dep', 'dep', 'zero', 'zero', 'zero', 'pad', 'pad', 'pad']]))
    assert score_all == 1/3
    assert score_dep == 1/3
    assert score_zero == 1/3
    assert list(tp) == [0, 1, 0, 0, 1, 0]
    assert list(fp) == [0, 2, 0, 0, 2, 0]
    assert list(fn) == [1, 0, 1, 1, 0, 1]

    score_all, score_dep, score_zero, tp, fp, fn = score_f(torch.tensor([[2,2,2,2,2,2,2,2,2]]), torch.tensor([[0,1,2,0,1,2,-1,-1,-1]]), np.array([['dep', 'dep', 'dep', 'zero', 'zero', 'zero', 'pad', 'pad', 'pad']]))
    assert score_all == 1/3
    assert score_dep == 1/3
    assert score_zero == 1/3
    assert list(tp) == [0, 0, 1, 0, 0, 1]
    assert list(fp) == [0, 0, 2, 0, 0, 2]
    assert list(fn) == [1, 1, 0, 1, 1, 0]


    label = torch.tensor([[ 3,  3,  0,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,
          3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,
          3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,
          3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  1,
          3,  3,  3,  3],
        [ 3,  3,  0,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,
          3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,
          3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,
          3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,
          3,  3,  3,  3],
        [ 3,  3,  3,  3,  3,  2,  3,  3,  3,  3,  3,  0,  3,  3,  3,  3,  3,  3,
          3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,
         -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
         -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
         -1, -1, -1, -1],
        [ 3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,
          3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  0,  3,  3,  3,  3,  3,
          3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,
          3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  2,  3,  3,  3,  3,  3,  3,  3,
          3,  3,  3,  3],
        [ 3,  3,  0,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,
          3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,
          3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,
          3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,
          3,  3,  3,  3],
        [ 3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,
          3,  3,  3,  0,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,
          3,  3,  3,  3,  3,  3,  3,  3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
         -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
         -1, -1, -1, -1],
        [ 3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  0,  3,  3,  3, -1,
         -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
         -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
         -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
         -1, -1, -1, -1],
        [ 3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,
          3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  0,  3,  3,  3,  3,  3,
         -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
         -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
         -1, -1, -1, -1]])
    pred = torch.tensor([[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
         3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
         3, 3, 3, 3, 0, 0, 0, 0, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 2, 2,
         0, 0, 0, 0],
        [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
         3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 3, 3, 3, 3, 3, 3,
         3, 3, 3, 3, 0, 0, 0, 2, 2, 2, 3, 0, 0, 0, 0, 0, 0, 3, 3, 3, 0, 0, 0, 0,
         0, 0, 0, 0],
        [1, 0, 3, 3, 2, 2, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
         2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
         2, 2, 2, 2],
        [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
         3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
         0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 0, 0, 0, 2, 0, 3, 3, 0, 0, 0, 0,
         0, 0, 0, 0],
        [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
         3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 3, 3,
         3, 0, 3, 3, 3, 0, 2, 0, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 3, 0,
         0, 0, 0, 0],
        [0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 2, 2, 2, 2,
         2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
         2, 2, 2, 2],
        [3, 3, 3, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2,
         2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
         2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
         2, 2, 2, 2],
        [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 3, 3, 3, 3, 0, 3, 3, 3, 0,
         0, 1, 1, 0, 0, 2, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
         2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
         2, 2, 2, 2]])
    prop = np.array([
        ['dep', 'dep', 'dep', 'dep', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'dep', 'dep', 'dep', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'dep', 'dep', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'dep', 'dep', 'pred', 'pred', 'pred'],
        ['zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'dep', 'dep', 'dep', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'dep', 'dep', 'pred', 'pred', 'pred', 'rentai', 'rentai', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero'],
        ['zero', 'zero', 'zero', 'zero', 'dep', 'dep', 'dep', 'pred', 'pred', 'pred', 'rentai', 'rentai', 'rentai', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad'],
        ['zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'dep', 'dep', 'dep', 'dep', 'pred', 'pred', 'pred', 'pred', 'rentai', 'rentai', 'zero', 'zero', 'zero'],
        ['zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'dep', 'dep', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'dep', 'dep', 'dep', 'pred', 'pred', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'rentai', 'rentai', 'rentai'],
        ['zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'dep', 'dep', 'dep', 'pred', 'pred', 'pred', 'rentai', 'rentai', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad'],
        ['zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'pred', 'zero', 'zero', 'rentai', 'rentai', 'zero', 'zero', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad'],
        ['zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'dep', 'dep', 'zero', 'zero', 'zero', 'dep', 'dep', 'pred', 'pred', 'pred', 'pred', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad'],
    ])

    score_all, score_dep, score_zero, tp, fp, fn = score_f(pred, label, prop)

    assert score_all == 0.05970149253731343
    assert score_dep == 0.19354838709677416
    assert score_zero == 0.0
    assert list(tp) == [4, 0, 2, 0, 0, 0]
    assert list(fp) == [40, 2, 6, 116, 9, 11]
    assert list(fn) == [1, 1, 0, 3, 0, 0]
    # precision_all = 6 / (6 + 184) = 3 / 85
    # precision_dep = 6 / (6 + 48) = 1 / 9
    # precision_zero = 0 / (6 + 184) = 0
    # recall_all = 6 / (6 + 5) = 6 / 11
    # recall_dep = 6 / (6 + 2) = 3 / 4
    # recall_zero = 0 / (6 + 3) = 0
    # f1_all =
