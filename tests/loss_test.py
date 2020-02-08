import losses.triplet_loss as l
import numpy as np
import scipy.spatial as sp
import torch
from torch.autograd import Variable
from losses.triplet_loss import topk, active
from losses.multi_loss import LinearWeightedLoss, WeightModule, MultiLoss, DynamicFocalLoss, DynamicFocalLossModule
from losses.regression import MSELoss, L1Loss, L2Loss
from dataflow import DataFlowConfig, DataFlowController

from losses.dummy import DummyLoss
import pytest

def test_cdist_different():
    a = np.random.randn(10, 20)
    b = np.random.randn(30, 20)

    for metric in ('euclidean', 'sqeuclidean', 'cityblock'):
        D_my = l.calc_cdist(torch.from_numpy(a), torch.from_numpy(b), metric).numpy()
        D_sp = sp.distance.cdist(a, b, metric)
        np.testing.assert_allclose(D_my, D_sp, rtol=1e-5, atol=1e-5)

def test_cdist_same():
    a = np.random.randn(10, 20)
    for metric in ('euclidean', 'sqeuclidean', 'cityblock'):
        D_my = l.calc_cdist(torch.from_numpy(a), torch.from_numpy(a), metric).numpy()
        D_sp = sp.distance.cdist(a, a, metric)
        np.testing.assert_allclose(D_my, D_sp, rtol=1e-5, atol=1e-5)

def test_active():
    x = np.array([0.1, 0.0, 0.3, 1, 0.4])
    x = torch.from_numpy(x)
    a = active(x)
    assert a == 4/5


@pytest.mark.parametrize("cuda", [True, False])
def test_batch_hard(cuda):
    pids = np.array([0, 0, 1, 0, 1, 1], dtype=np.float32)
    features = np.array([
        [5.0],
        [6.0],
        [1.0],
        [7.0],
        [9.5],
        [1.0]
    ], np.float32)

    pids = Variable(torch.from_numpy(pids))
    data = {'pid': pids}
    features = torch.from_numpy(features)
    if cuda:
        features = features.cuda()
    features = Variable(features)

    loss_fn = l.BatchHard("none")
    endpoints = {"triplet": [features]}
    loss = loss_fn(endpoints, data)

    result = np.array([2.0 - 4.0, 1.0 - 3.5, 8.5 - 4.0, 2.0 - 2.5, 8.5 - 2.5, 8.5 - 4.0], dtype=np.float32)
    if cuda:
        loss = loss.data.cpu().numpy()
    else:
        loss = loss.data.numpy()
    np.testing.assert_array_equal(result, loss)


def test_topk():
    pids = np.array([0, 0, 1, 0, 1], dtype=np.float32)
    features = np.array([
        [5.0],
        [6.0],
        [1.0],
        [7.0],
        [9.5],
    ], np.float32)

    pids = Variable(torch.from_numpy(pids))
    features = Variable(torch.from_numpy(features))
    cdist = l.calc_cdist(features, features)
    topks = topk(cdist, pids, 4)
    np.testing.assert_almost_equal(topks[0], 3/5)
    np.testing.assert_almost_equal(topks[1], 3/5)
    np.testing.assert_almost_equal(topks[2], 3/5)
    np.testing.assert_almost_equal(topks[3], 5/5)


def test_weighted_loss():
    endpoint_name = "dummy"
    weight = LinearWeightedLoss(0.5, DummyLoss())
    losses = {"loss1": weight, "loss2": weight}
    linear_weighted_loss = WeightModule(losses)
    features_np = np.array([
        [5.0],
        [6.0],
        [1.0],
        [7.0],
        [9.5],
    ], np.float32)
    features = torch.from_numpy(features_np)
    endpoints = {endpoint_name: [features]}
    split_endpoints = {"loss1": endpoints, "loss2": endpoints}
    split_data = {"loss1": None, "loss2": None}
    loss = linear_weighted_loss(split_endpoints, split_data)
    np.testing.assert_array_equal(loss, np.sum(features_np))


def test_dynamic_focal_loss():
    endpoint_name = "dummy"
    weight = DynamicFocalLoss(1, 1, 1e-6, DummyLoss(), "dynamic")
    id_loss =("loss1", weight)
    tr_loss = ("loss2", weight)
    dynamic_focal_loss = DynamicFocalLossModule(0, tr_loss, id_loss)
    split_data = {"loss1": None, "loss2": None}
    for i in range(10):
        features = np.random.rand(10, 1) * 10
        features = features.astype(np.float32)
        features = torch.from_numpy(features)
        endpoints = {endpoint_name: [features]}
        split_endpoints = {"loss1": endpoints, "loss2": endpoints}
        loss = dynamic_focal_loss(split_endpoints, split_data)


test_cfgs = [
        DataFlowConfig("dummy1", "loss"),
        DataFlowConfig("dummy2", "loss"),
        DataFlowConfig("all", "loss"),
        DataFlowConfig(["dummy1", "dummy2"], "loss")
    ]


@pytest.mark.parametrize("cfg", test_cfgs)
def test_multi_loss(cfg):
    head = "dummy"
    weight = LinearWeightedLoss(1, DummyLoss())
    losses = {"loss": weight}
    linear_weighted_loss = WeightModule(losses)

    data_controller = DataFlowController([cfg])
    multi_loss = MultiLoss(linear_weighted_loss, data_controller)
    features_np = np.array([
        [5.0],
        [6.0],
        [1.0],
        [7.0],
        [9.5],
    ], np.float32)
    features = torch.from_numpy(features_np)
    endpoints = {head: [features]}

    idxs1 = [0, 1, 2]
    idxs2 = [3, 4]
    dataset1 = "dummy1"
    dataset2 = "dummy2"
    split_info = {
        dataset1: idxs1,
        dataset2: idxs2
    }
    data = {'split_info': split_info}

    loss = multi_loss(endpoints, data)
    if cfg.targets[0] == "all":
        np.testing.assert_array_equal(loss, np.sum(features_np))
    else:
        correct = 0
        for d in cfg.targets:
            idxs = split_info[d]
            correct += features[idxs].sum()
        print(loss, correct)
        np.testing.assert_array_equal(loss, correct)


def n2t(array):
    return torch.from_numpy(array)


def test_mse_loss():
    loss = MSELoss('l2', 'target')
    l2 = np.array(
            [[1.0, 0.0],
             [0.0, 1.0],
             [1.0, 1.0],
             [0.0, 0.0]], dtype=np.float32)
    endpoints = {'l2': n2t(l2)}
    target = np.array(
            [[0.0, 0.0],
             [1.0, 1.0],
             [np.nan, np.nan],
             [0.0, 0.0]], dtype=np.float32)
    data = {'target': n2t(target)}

    result = loss(endpoints, data)
    np.testing.assert_array_equal(result.numpy(), np.array(2/6, dtype=np.float32))


def test_l2_loss():
    loss = L2Loss('l2', 'target')
    l2 = np.array(
            [[[1.0, 0.0],
             [0.0, 1.0],
             [1.0, 1.0],
             [0.0, 0.0]]], dtype=np.float32)
    endpoints = {'l2': n2t(l2)}
    target = np.array(
            [[[0.0, 0.0],
             [1.0, 1.0],
             [np.nan, np.nan],
             [0.0, 0.0]]], dtype=np.float32)
    data = {'target': n2t(target)}

    result = loss(endpoints, data)
    np.testing.assert_array_equal(result.numpy(), np.array(2/3, dtype=np.float32))

from losses.softmax import get_topk_percent
def test_get_topk_percent():
    tensor = torch.tensor([[[4, 3, 8, 2], [8, 5, 1, -1]]])
    values, indices = get_topk_percent(tensor, 0.5)
    np.testing.assert_array_equal(values.numpy(), np.array([[8, 8, 5, 4]]))

