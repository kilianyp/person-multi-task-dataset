from writers.h5 import write_to_h5
from torch.nn.parallel import DataParallel
import builders.dataset_builder as dataset_builder
import builders.evaluation_model_builder as model_cfg_builder
import builders.model_builder as model_builder
import torch
import time
import tempfile
import objgraph


transform_cfg = {
    "resize": {
        "width": 128,
        "height": 256
    },
    "normalization": {
        "mean": [0.485, 0.456, 0.406],
        "std" : [0.229, 0.224, 0.225]
    }
}
gallery_cfg = {
    "transform": transform_cfg,
    "source_file": "/home/pfeiffer/Projects/cupsizes/data/market1501_test.csv",
    "data_dir": "/home/pfeiffer/Projects/triplet-reid-pytorch/data/Market-1501",
    "loader_fn": "pil",
    "name": "reid_dataset1",
    "type": "reid",
    "limit": 1000
}
query_cfg = {
    "transform": transform_cfg,
    "source_file": "/home/pfeiffer/Projects/cupsizes/data/market1501_query.csv",
    "data_dir": "/home/pfeiffer/Projects/triplet-reid-pytorch/data/Market-1501",
    "loader_fn": "pil",
    "name": "reid_dataset1",
    "type": "reid",
    "limit": 1006
}

cfg ={
    "files": ["/work/pfeiffer/master/attributes/57/model_300"]
        }

def test_reid_writing():
    gallery_dataset = dataset_builder.build(gallery_cfg)
    query_dataset = dataset_builder.build(gallery_cfg)
    gallery_dataloader = torch.utils.data.DataLoader(
            gallery_dataset,
            batch_size=10,
            num_workers=0
    )
    model_cfg = model_cfg_builder.build(cfg)
    model = model_builder.build(model_cfg[0])
    model = DataParallel(model)
    output_file = tempfile.mktemp()
    objgraph.show_growth()
    write_to_h5(gallery_dataloader, model, output_file)
    objgraph.show_most_common_types()
    objgraph.show_growth()
    time.sleep(1)
    query_dataloader = torch.utils.data.DataLoader(
            query_dataset,
            batch_size=10,
            num_workers=0
    )
    output_file = tempfile.mktemp()
    write_to_h5(query_dataloader, model, output_file)


pose_dataset_cfg = {
    "name": "mpii",
    "source_file": "/work/pfeiffer/datasets/mpii/mpii_human_pose_v1_u12_1.mat",
    "data_dir": "/fastwork/pfeiffer/mpii/",
    "kwargs": {
        "split": "val"
    },
    "dataset_fn": "mpii",
    "loader_fn": "cv2",
    "transform": {
        "resize": {
            "width": 256,
            "height": 256
        },
        "normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std" : [0.229, 0.224, 0.225]
        }
    },
    "type": "pose"
}


pose_model_cfg ={
    "files": ["/work/pfeiffer/master/pose/124/model_80"]
}


def test_pose_writing():
    pose_dataset = dataset_builder.build(pose_dataset_cfg)
    pose_dataloader = torch.utils.data.DataLoader(
            pose_dataset,
            batch_size=10,
            num_workers=0
    )
    model_cfg = model_cfg_builder.build(pose_model_cfg)[0]
    dataset_info = pose_dataset.info
    new_cfg = dataset_info.copy()
    new_cfg.update(model_cfg)
    print(new_cfg)
    model = model_builder.build(new_cfg)
    model = DataParallel(model)
    output_file = './tests/pose.h5'
    write_to_h5(pose_dataloader, model, output_file, ['emb', 'pose'])
