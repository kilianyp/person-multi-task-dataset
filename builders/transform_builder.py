import builders.imgaug_transform_builder as imgaug_transform_builder
import builders.torchvision_transform_builder as torchvision_transform_builder


def build(cfg, dataset_info={}):
    backend = cfg.get('backend', 'imgaug').lower()
    if backend == "imgaug":
        return imgaug_transform_builder.build(cfg, dataset_info)
    elif backend == "torchvision":
        return torchvision_transform_builder.build(cfg, dataset_info)

    raise ValueError("Unknown transform backend {}.".format(backend))
