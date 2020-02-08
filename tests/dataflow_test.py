from dataflow import DataFlowConfig, DataFlowController
import numpy as np


def test_data_flow_config():
    output_name1 = "loss1"
    output_name2 = "loss2"
    cfg1 = DataFlowConfig(targets="dataset1", output_name=output_name1)
    cfg2 = DataFlowConfig(targets="dataset2", output_name=output_name2)

    controller = DataFlowController([cfg1, cfg2])

    data = np.arange(20)

    split_info = {"dataset1": np.arange(10), "dataset2": np.arange(10) + 10}

    splits = controller.split(data, split_info)

    np.testing.assert_array_equal(splits[output_name1], np.arange(10))
    np.testing.assert_array_equal(splits[output_name2], np.arange(10) + 10)




