import numpy as np
from datetime import datetime

def calculate_pckh(dist, head_sizes, thresholds=[0.1, 0.5, 1]):
    # only visible
    non_visible = np.isnan(dist)
    visible = 1 - non_visible
    # PCKh, norm by headsize
    error = dist/head_sizes[:, None]

    # ignore nan
    ALMOST_INF = 9999
    error[non_visible] = ALMOST_INF

    # visible joints per joint
    jnt_count_joint = np.sum(visible, axis=0)


    pck_all = {}
    pck_joint = {}
    for threshold in thresholds:
        pass_threshold = (error <= threshold).astype(np.int32)
        per_joint = np.sum(pass_threshold, axis=0) / jnt_count_joint
        pck_joint[threshold] = per_joint
        pck_all[threshold] = pass_threshold.sum() / visible.sum()

    return pck_all, pck_joint


def average_precision_score(y_true, y_score):
    """ From pull request #45
    https://github.com/VisualComputingInstitute/triplet-reid/pull/45
    
    Compute average precision (AP) from prediction scores.
    This is a replacement for the scikit-learn version which, while likely more
    correct does not follow the same protocol as used in the default Market-1501
    evaluation that first introduced this score to the person ReID field.
    Args:
        y_true (array): The binary labels for all data points.
        y_score (array): The predicted scores for each samples for all data
            points.
    Raises:
        ValueError if the length of the labels and scores do not match.
    Returns:
        A float representing the average precision given the predictions.
    """

    if len(y_true) != len(y_score):
        raise ValueError('The length of the labels and predictions must match '
                         'got lengths y_true:{} and y_score:{}'.format(
                            len(y_true), len(y_score)))

    y_true_sorted = y_true[np.argsort(-y_score, kind='mergesort')]

    tp = np.cumsum(y_true_sorted)
    total_true = np.sum(y_true_sorted)
    recall = tp / total_true
    recall = np.insert(recall, 0, 0.)
    precision = tp / np.arange(1, len(tp) + 1)
    precision = np.insert(precision, 0, 1.)
    ap = np.sum(np.diff(recall) * ((precision[1:] + precision[:-1]) / 2))

    return ap


def calc_euclidean(array1, array2):
    """
    Calc euclidean from
    https://github.com/huanghoujing/person-reid-triplet-loss-baseline/blob/master/tri_loss/utils/distance.py
    """
    # shape [m1, 1]
    square1 = np.sum(np.square(array1), axis=1)[..., np.newaxis]
    # shape [1, m2]
    square2 = np.sum(np.square(array2), axis=1)[np.newaxis, ...]
    squared_dist = - 2 * np.matmul(array1, array2.T) + square1 + square2
    squared_dist[squared_dist < 0] = 0
    dist = np.sqrt(squared_dist)
    return dist


def reid_score(dist_mat,
              query_pids, query_fids, 
              gallery_pids, gallery_fids,
              matcher):
    """
    Compute mAP and CMC score.

    Adapted from
    https://github.com/VisualComputingInstitute/triplet-reid/
    """
    cmc = np.zeros(len(gallery_pids), dtype=np.int32)
    aps = []

    # Compute the pid matches
    pid_matches = gallery_pids[None] == query_pids[:,None]
    # Compute the mask using the dataset specific mask function.
    mask = matcher(query_fids)

    # Modify the distances and pid matches.
    dist_mat[mask] = np.inf
    pid_matches[mask] = False

    # Keep track of statistics
    scores = 1 / (1 + dist_mat)

    for i in range(len(dist_mat)):
        ap = average_precision_score(pid_matches[i], scores[i])

        if np.isnan(ap):
            print()
            print("WARNING: encountered an AP of NaN!")
            print("This usually means a person only appears once.")
            print("In this case, it's because of {}.".format(query_fids[i]))
            print("I'm excluding this person from eval and carrying on.")
            print()
            continue

        aps.append(ap)
        # Find the first true match and increment the cmc data.
        k = np.where(pid_matches[i, np.argsort(dist_mat[i])])[0][0]
        cmc[k:] += 1

    # Compute the actual cmc and mAP values
    cmc = cmc / len(query_pids)
    mean_ap = np.mean(aps)
    print('mAP: {:.2%} | top-1: {:.2%} | top-5: {:.2%} | top-10: {:.2%}'.format(
        mean_ap, cmc[0], cmc[4], cmc[9]))

    return mean_ap, cmc


"""
https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/score.py
"""
def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)


def compute_hist(gts, preds, num_classes):
    n_cl = num_classes
    hist = np.zeros((n_cl, n_cl))
    for gt, pred in zip(gts, preds):
        hist += fast_hist(gt.flatten(), pred.flatten(), n_cl)
    return hist


def calc_seg_score(hist, id_to_label=None):
    score = {}
    acc = np.diag(hist).sum() / hist.sum()
    score['overall_accuracy'] = acc
    # per-class accuracy
    acc = np.diag(hist) / hist.sum(1)
    score['mean_accuracy'] = np.nanmean(acc)
    # per-class IU
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    if id_to_label is None:
        for idx, per_class_iou in enumerate(iu):
            score['class_{}_iou'.format(idx)] = per_class_iou
    else:
        for idx, per_class_iou in enumerate(iu):
            score['class_{}_iou'.format(id_to_label[idx])] = per_class_iou
    score['miou'] = np.nanmean(iu)
    freq = hist.sum(1) / hist.sum()
    score['fwavacc'] = (freq[freq > 0] * iu[freq > 0]).sum()
    #score['hist'] = hist
    return score
