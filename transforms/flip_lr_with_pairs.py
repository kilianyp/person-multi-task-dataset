from imgaug.augmenters import Fliplr
import imgaug as ia


def flip_joints(pairs, keypoints, width):
    for keypoint in keypoints:
        # copy logic from imgaug
        keypoint.x = (width - 1) - keypoint.x

    for (joint1, joint2) in pairs:
        # Swap
        keypoints[joint1], keypoints[joint2] = keypoints[joint2], keypoints[joint1]

import warnings
class FliplrWithPairs(Fliplr):
    # TODO this should also flip the bbox
    def __init__(self, *args, keypoint_pairs=None, segmentation_pairs=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.keypoint_pairs = keypoint_pairs
        self.segmentation_pairs = segmentation_pairs
        # WARNING to_deterministic calls copy
        # it seems like the functions are not copied as intended, the augment segmentation
        # maps function then has still the self reference to the old object
        #self.augment_segmentation_maps = self._augment_segmentation_maps_with_pairs
        warnings.warn("FliplrWithPairs does not work in Pipeline for segmentation masks!.")

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        if self.keypoint_pairs is None:
            return super()._augment_keypoints(keypoints_on_images, random_state, parents, hooks)
        nb_images = len(keypoints_on_images)
        samples = self.p.draw_samples((nb_images,), random_state=random_state)
        for i, keypoints_on_image in enumerate(keypoints_on_images):
            if samples[i] == 1:
                width = keypoints_on_image.shape[1]
                keypoints = keypoints_on_image.keypoints
                flip_joints(self.keypoint_pairs, keypoints, width)
        return keypoints_on_images


    def augment_segmentation_maps(self, segmaps, parents=None, hooks=None):
        """
        Switches the segmentation idexes of the non empty maps for the heatmaps.
        This will lead to a correct switching during the building of the segmap.

        we need to get the random state and check if they were switched.
        """
        if self.segmentation_pairs is None:
            return super().augment_segmentation_maps(segmaps, parents, hooks)

        input_was_single_instance = False
        if isinstance(segmaps, ia.SegmentationMapOnImage):
            input_was_single_instance = True
            segmaps = [segmaps]

        # to heatmaps only_nonempty=True returns only maps of nonempty classes
        # the 2nd return value is a list of nonempty class indices
        heatmaps_with_nonempty = [segmap.to_heatmaps(only_nonempty=True, not_none_if_no_nonempty=True)
                                  for segmap in segmaps]
        heatmaps = [heatmaps_i for heatmaps_i, nonempty_class_indices_i in heatmaps_with_nonempty]
        nonempty_class_indices = [nonempty_class_indices_i
                                  for heatmaps_i, nonempty_class_indices_i in heatmaps_with_nonempty]
        heatmaps_aug = self.augment_heatmaps(heatmaps, parents=parents, hooks=hooks)
        segmaps_aug = []

        nb_images = len(segmaps)

        # save original state, cmp to augment_images
        if self.deterministic:
            state_orig = self.random_state.get_state()
        samples = self.p.draw_samples((nb_images,), random_state=self.random_state)
        if self.deterministic:
            self.random_state.set_state(state_orig)

        for segmap, heatmaps_aug_i, nonempty_class_indices_i in zip(segmaps, heatmaps_aug, nonempty_class_indices):
            if samples[0] == 1:
                switched_indices_i = []
                for idx in nonempty_class_indices_i:
                    if idx in self.segmentation_pairs:
                        idx = self.segmentation_pairs[idx]
                    switched_indices_i.append(idx)
                class_indices = switched_indices_i
            else:
                class_indices = nonempty_class_indices_i
            segmap_aug = ia.SegmentationMapOnImage.from_heatmaps(heatmaps_aug_i,
                                                                 class_indices=class_indices,
                                                                 nb_classes=segmap.nb_classes)
            segmap_aug.input_was = segmap.input_was
            segmaps_aug.append(segmap_aug)

        if input_was_single_instance:
            return segmaps_aug[0]
        return segmaps_aug, heatmaps_aug


