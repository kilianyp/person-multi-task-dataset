from datasets.attribute_dataset import AttributeDataset
from datasets.attribute_reid_dataset import AttributeReidDataset, AttributeReidTestDataset
from datasets import register_dataset
import numpy as np
from datasets.utils import read_from_mat
from settings import Config
from datasets.reid.market import MarketReid, MarketReidTest


market_attribute_header = ['gender', 'hair', 'up', 'down', 'clothes', 'hat', 'backpack', 'bag', 'handbag', 'age', 'upcolor', 'downcolor']
def make_market_attribute(path, split):
    if split == 'train':
        num_pids = 751
    elif split == 'test':
        num_pids = 750
    else:
        raise ValueError(split)
    upcolors = ['upblack', 'upwhite', 'upred', 'uppurple', 'upyellow', 'upgray', 'upblue', 'upgreen']
    downcolors = ['downblack', 'downwhite', 'downpink', 'downpurple', 'downyellow', 'downgray', 'downblue', 'downgreen', 'downbrown']

    def get_upcolor(row):
        # upcolor is col 10-17 (8 values)
        return np.argmax(row[10:18])

    def get_downcolor(row):
        # downcolor is col 18-26 (9 values)
        downcolors = row[18:27]
        return np.argmax(downcolors)

    pid_data = read_from_mat(path)['market_attribute']
    mat_struct = pid_data[split][0, 0]
    attributes = np.ndarray((num_pids, 27), dtype=np.int32)
    # Note colors are all the same
    for idx, h in enumerate(market_attribute_header[:-2]):
        col = mat_struct[h][0, 0]
        attributes[:, idx] = col[:]

    for idx, h in enumerate(upcolors):
        idx += len(market_attribute_header) - 2
        # print(idx, h)
        col = mat_struct[h][0, 0]
        attributes[:, idx] = col[:]

    for idx, h in enumerate(downcolors):
        idx += len(market_attribute_header) - 2 + len(upcolors)
        # print(idx, h)
        col = mat_struct[h][0, 0]
        attributes[:, idx] = col[:]
    data = []
    for row in attributes:
        d = {}
        for idx, h in enumerate(market_attribute_header):
            if h == "upcolor":
                value = get_upcolor(row)
            elif h == "downcolor":
                value = get_downcolor(row)
            else:
                value = row[idx] - 1
            d[h] = value
        data.append(d)
    # set default header
    headers = {}
    # dont write headers
    # TODO very confusing, this is using the data from memory.
    # Problem is that endpoints and data again have same name
    dataset_info = {}
    dataset_info['attributes'] = market_attribute_header
    return data, headers, dataset_info


@register_dataset("market1501_attribute")
class MarketAttribute(AttributeDataset):
    def __init__(self, data, headers, info):
        super().__init__("market1501", data, headers, info)

    @staticmethod
    def build(cfg, *args, **kwargs):
        split = cfg['split']
        source_file = Config.MARKET_ATTRIBUTE
        data, headers, info = make_market_attribute(source_file, split)
        return MarketAttribute(data, headers, info)


@register_dataset("market1501_attribute_reid")
class MarketAttributeReid(AttributeReidDataset):
    @staticmethod
    def build(cfg, *args, **kwargs):
        attribute = MarketAttribute.build(cfg['attribute'], *args, **kwargs)
        reid = MarketReid.build(cfg['reid'], *args, **kwargs)
        return MarketAttributeReid(attribute, reid)


@register_dataset("market1501_attribute_reid_test")
class MarketAttributeReidTest(AttributeReidTestDataset):
    @staticmethod
    def build(cfg, *args, **kwargs):
        attribute = MarketAttribute.build(cfg['attribute'], *args, **kwargs)
        reid = MarketReidTest.build(cfg['reid'], *args, **kwargs)
        return MarketAttributeReidTest(attribute, reid)
