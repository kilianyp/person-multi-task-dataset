from datasets.attribute_dataset import AttributeDataset
from datasets.attribute_reid_dataset import AttributeReidDataset, AttributeReidTestDataset
from datasets import register_dataset
import numpy as np
from datasets.utils import read_from_mat
from logger import get_logger
from settings import Config
from datasets.reid.duke_mtmc import DukeMTMCReid, DukeMTMCReidTest

duke_attribute_header = ['gender', 'top', 'boots', 'hat', 'backpack', 'bag', 'handbag', 'shoes', 'upcolor', 'downcolor']
def make_duke_attribute(path, split):
    logger = get_logger()
    if split == 'train':
        num_pids = 702
    elif split == 'test':
        num_pids = 1110
    else:
        raise ValueError
    upcolors = ['upblack', 'upwhite', 'upred', 'uppurple', 'upgray', 'upblue', 'upgreen', 'upbrown']
    downcolors = ['downblack', 'downwhite', 'downred', 'downgray', 'downblue', 'downgreen', 'downbrown']

    def get_upcolor(row):
        # upcolor is col 8-15 (8 values)
        return np.argmax(row[8:16])

    def get_downcolor(row):
        # downcolor is col 16-22 (7 values)
        downcolors = row[16:23]
        return np.argmax(downcolors)

    pid_data = read_from_mat(path)['duke_attribute']
    mat_struct = pid_data[split][0, 0]
    attributes = np.ndarray((num_pids, 24), dtype=np.int32)
    # Note colors are all the same
    for idx, h in enumerate(duke_attribute_header[:-2]):
        col = mat_struct[h][0, 0]
        attributes[:, idx] = col[:]

    for idx, h in enumerate(upcolors):
        idx += len(duke_attribute_header) - 2
        # print(idx, h) col = mat_struct[h][0, 0] attributes[:, idx] = col[:]

    for idx, h in enumerate(downcolors):
        idx += len(duke_attribute_header) - 2 + len(upcolors)
        # print(idx, h)
        col = mat_struct[h][0, 0]
        attributes[:, idx] = col[:]
    data = []
    for pid, row in enumerate(attributes):
        d = {}
        for idx, h in enumerate(duke_attribute_header):
            if h == "upcolor":
                value = get_upcolor(row)
            elif h == "downcolor":
                value = get_downcolor(row)
            else:
                value = row[idx] - 1
                if value < 0:
                    logger.warning("Undefined value for pid %d and attribute %s!", pid, h)
                    # TODO for duke this happens for two pids, how to handle properly
                    value = 0
            d[h] = value
        data.append(d)
    # set default header
    headers = {}
    dataset_info = {}
    # dont write headers
    # TODO very confusing, this is using the data from memory.
    # Problem is that endpoints and data again have same name
    dataset_info['attributes'] = duke_attribute_header
    return data, headers, dataset_info


@register_dataset("duke_mtmc_attribute")
class DukeAttribute(AttributeDataset):
    def __init__(self, data, headers, info):
        super().__init__("duke_mtmc", data, headers, info)

    @staticmethod
    def build(cfg, *args, **kwargs):
        split = cfg['split']
        source_file = Config.MARKET_ATTRIBUTE
        data, headers, info = make_duke_attribute(source_file, split)
        return DukeAttribute(data, headers, info)

@register_dataset("duke_mtmc_attribute_reid")
class DukeAttributeReid(AttributeReidDataset):
    @staticmethod
    def build(cfg, *args, **kwargs):
        attribute = DukeAttribute.build(cfg['attribute'])
        reid = DukeMTMCReid.build(cfg['reid'], *args, **kwargs)
        return DukeAttributeReid(attribute, reid)


@register_dataset("duke_mtmc_attribute_reid_test")
class DukeAttributeReidTest(AttributeReidTestDataset):
    @staticmethod
    def build(cfg, *args, **kwargs):
        attribute = DukeAttribute.build(cfg['attribute'])
        reid = DukeMTMCReidTest.build(cfg['reid'], *args, **kwargs)
        return DukeAttributeReid(attribute, reid)

