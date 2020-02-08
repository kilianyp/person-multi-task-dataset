import numpy as np
import imgaug.augmenters.meta as meta
from imgaug import parameters as iap
import imgaug as ia
from .utils import (transform_from_bbox, transform_and_crop_image,
                    trans_point2d)


class AffineWithCrop(meta.Augmenter):
    """
    Apply an affine augmentation and Cropping on an Image.
    Relative augmenations are done with respect to the bbox.

    TODO cannot declare properties.
    """
    def __init__(self, scale=1.0, translate_percent=None, rotate=0.0, name=None, deterministic=False, random_state=None):
        """Copied from imgaug/transforms/geometric.py."""
        super().__init__(name=name, deterministic=deterministic, random_state=random_state)
        # scale
        if isinstance(scale, dict):
            ia.do_assert("x" in scale or "y" in scale)
            x = scale.get("x", 1.0)
            y = scale.get("y", 1.0)
            self.scale = (
                iap.handle_continuous_param(x, "scale['x']", value_range=(0+1e-4, None), tuple_to_uniform=True,
                                            list_to_choice=True),
                iap.handle_continuous_param(y, "scale['y']", value_range=(0+1e-4, None), tuple_to_uniform=True,
                                            list_to_choice=True)
            )
        else:
            self.scale = iap.handle_continuous_param(scale, "scale", value_range=(0+1e-4, None), tuple_to_uniform=True,
                                                     list_to_choice=True)

        # translate by percent
        if isinstance(translate_percent, dict):
            ia.do_assert("x" in translate_percent or "y" in translate_percent)
            x = translate_percent.get("x", 0)
            y = translate_percent.get("y", 0)
            self.translate = (
                iap.handle_continuous_param(x, "translate_percent['x']", value_range=None, tuple_to_uniform=True,
                                            list_to_choice=True),
                iap.handle_continuous_param(y, "translate_percent['y']", value_range=None, tuple_to_uniform=True,
                                            list_to_choice=True)
            )
        else:
            self.translate = iap.handle_continuous_param(translate_percent, "translate_percent", value_range=None,
                                                             tuple_to_uniform=True, list_to_choice=True)
        self.rotate = iap.handle_continuous_param(rotate, "rotate", value_range=None, tuple_to_uniform=True,
                                                  list_to_choice=True)

        self.centers = None
        self.bboxs = None

    def _augment_images(self, images, random_state, parents, hooks):
        nb_images = len(images)
        ia.do_assert(len(self.bboxs) == nb_images)
        ia.do_assert(len(self.centers) == nb_images)
        scale_samples, translate_samples, rotate_samples = self._draw_samples(nb_images, random_state)
        augmented = []
        # Translation is currently not used
        for i, image in enumerate(images):
            bbox = self.bboxs[i]
            bb_width, bb_height = bbox
            c_x, c_y = self.centers[i]
            scale = scale_samples[i]
            rotate = rotate_samples[i]
            translate = translate_samples[i]
            if len(translate) == 1:
                trans_x = trans_y = translate[0]
            elif len(translate) == 2:
                trans_x, trans_y = translate
            else:
                raise ValueError
            trans = transform_from_bbox(c_x, c_y, bb_width, bb_height, scale, rotate, trans_x, trans_y)
            image = transform_and_crop_image(image, trans, bbox)
            augmented.append(image)
        return augmented

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        nb_images = len(keypoints_on_images)
        ia.do_assert(len(self.bboxs) == nb_images)
        scale_samples, translate_samples, rotate_samples = self._draw_samples(nb_images, random_state)
        augmented = []
        for i, keypoints_on_image in enumerate(keypoints_on_images):
            bb_width, bb_height = self.bboxs[i]
            c_x, c_y = self.centers[i]
            scale = scale_samples[i]
            rotate = rotate_samples[i]
            translate = translate_samples[i]
            if len(translate) == 1:
                trans_x = trans_y = translate[0]
            elif len(translate) == 2:
                trans_x, trans_y = translate
            else:
                raise ValueError
            trans = transform_from_bbox(c_x, c_y, bb_width, bb_height, scale, rotate, trans_x, trans_y)
            augmented_keypoints = trans_point2d(keypoints_on_image.get_coords_array(), trans)
            augmented_keypoints = ia.KeypointsOnImage.from_coords_array(augmented_keypoints, shape=(bb_height, bb_width))
            augmented.append(augmented_keypoints)
        return augmented

    def _augment_heatmaps(self, heatmaps, random_state, parents, hooks):
        raise NotImplementedError

    def get_parameters(self):
        return [self.scale, self.translate, self.rotate]

    def _draw_samples(self, nb_samples, random_state):
        seed = random_state.randint(0, 10**6, 1)[0]

        if isinstance(self.scale, tuple):
            scale_samples = (
                self.scale[0].draw_samples((nb_samples,), random_state=ia.new_random_state(seed + 10)),
                self.scale[1].draw_samples((nb_samples,), random_state=ia.new_random_state(seed + 20)),
            )
        else:
            scale_samples = self.scale.draw_samples((nb_samples,), random_state=ia.new_random_state(seed + 30))
            scale_samples = (scale_samples, scale_samples)

        if isinstance(self.translate, tuple):
            translate_samples = (
                self.translate[0].draw_samples((nb_samples,), random_state=ia.new_random_state(seed + 40)),
                self.translate[1].draw_samples((nb_samples,), random_state=ia.new_random_state(seed + 50)),
            )
        else:
            translate_samples = self.translate.draw_samples((nb_samples,), random_state=ia.new_random_state(seed + 60))
            translate_samples = (translate_samples, translate_samples)

        ia.do_assert(translate_samples[0].dtype in [np.int32, np.int64, np.float32, np.float64])
        ia.do_assert(translate_samples[1].dtype in [np.int32, np.int64, np.float32, np.float64])

        rotate_samples = self.rotate.draw_samples((nb_samples,), random_state=ia.new_random_state(seed + 70))

        return scale_samples, translate_samples, rotate_samples


    def update(self, centers, bboxs):
        """Affine needs to update during each step.

        TODO can this be somehow hidden that the interface between
        crop and no crop datasets is the same?
        """
        self.centers = centers
        self.bboxs = bboxs

