import imgaug.augmenters.meta as meta
import imgaug as ia
from .utils import (transform_from_bbox, transform_and_crop_image,
                    trans_point2d)


class CropFromBbox(meta.Augmenter):
    """
    Apply an affine augmentation and Cropping on an Image.

    TODO cannot assign properties.
    """
    def __init__(self, name=None, deterministic=False, random_state=None):
        """Copied from imgaug/transforms/geometric.py."""
        super().__init__(name=name, deterministic=deterministic, random_state=random_state)

        self.centers = None
        self.bboxs = None

    def _augment_images(self, images, random_state, parents, hooks):
        nb_images = len(images)
        ia.do_assert(len(self.bboxs) == nb_images)
        ia.do_assert(len(self.centers) == nb_images)
        augmented = []
        # Translation is currently not used
        for i, image in enumerate(images):
            bbox = self.bboxs[i]
            bb_width, bb_height = bbox
            c_x, c_y = self.centers[i]
            trans = transform_from_bbox(c_x, c_y, bb_width, bb_height, 1.0, 0, 0, 0)
            image = transform_and_crop_image(image, trans, bbox)
            augmented.append(image)
        return augmented

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        nb_images = len(keypoints_on_images)
        ia.do_assert(len(self.bboxs) == nb_images)
        augmented = []
        for i, keypoints_on_image in enumerate(keypoints_on_images):
            bb_width, bb_height = self.bboxs[i]
            c_x, c_y = self.centers[i]
            trans = transform_from_bbox(c_x, c_y, bb_width, bb_height, 1.0, 0, 0, 0)
            augmented_keypoints = trans_point2d(keypoints_on_image.get_coords_array(), trans)
            augmented_keypoints = ia.KeypointsOnImage.from_coords_array(augmented_keypoints, shape=(bb_height, bb_width))
            augmented.append(augmented_keypoints)
        return augmented

    def _augment_heatmaps(self, heatmaps, random_state, parents, hooks):
        raise NotImplementedError

    def get_parameters(self):
        return []

    def update(self, centers, bboxs):
        """Affine needs to update during each step.

        TODO can this be somehow hidden that the interface between
        crop and no crop datasets is the same?
        """
        self.centers = centers
        self.bboxs = bboxs

