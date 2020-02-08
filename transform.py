import imgaug as ia
from abc import abstractmethod, ABC
from torchvision import transforms as tvtransforms

class Transform(ABC):
    def __init__(self, transform, normalize, debug, num_copies=1):
        self.debug = debug
        self.transform = transform
        self.normalize = normalize
        self.to_tensor = tvtransforms.ToTensor()
        self.num_copies = num_copies

    @abstractmethod
    def augment_images(self, images, return_unnormalized=False):
        pass

    @abstractmethod
    def augment_image(self, image, return_unnormalized=False):
        pass

    @abstractmethod
    def augment_keypoints(self, keypoints, shape):
        pass

    @abstractmethod
    def augment_keypoint(self, keypoints, shape):
        pass

    @abstractmethod
    def augment_segmentation(self, image, num_classes):
        pass

    @abstractmethod
    def to_deterministic(self):
        pass


def _transform_imgaug(self, image):

    aug_image = self.det_transform.augment_image(image)
    # TODO can be removed if normalize supports negative stride
    # Does not support test time transform
    aug_image = aug_image.copy()

    return aug_image


def _transform_torchvision(self, image):
    aug_image = self.transform(image)
    if isinstance(aug_image, tuple):
        # some augmentations return two images as a tuple
        # copy does not exist
        # TODO negative stride will still fail
        # Does this error actually occur with torchvision?
        pass
    else:
        # Copy is needed due to negative strides
        aug_image = aug_image.copy()

    return aug_image


class ImgAugTransform(Transform):
    def __init__(self, transform, normalize, debug):
        """
        """
        super().__init__(transform, normalize, debug)
        self.det_transform = None

    def augment_image(self, image, return_unnormalized=False):
        aug_image = _transform_imgaug(self, image)
        if return_unnormalized:
            aug_image_unnorm = aug_image.copy()

        if not self.debug:
            aug_image = self.normalize(aug_image)

        if return_unnormalized:
            return aug_image, aug_image_unnorm
        else:
            return aug_image

    def augment_images(self, images, return_unnormalized=False):
        raise NotImplementedError

    def augment_keypoint(self, keypoints, shape):
        keypoints_on_image = ia.KeypointsOnImage.from_coords_array(keypoints, shape=shape)
        return self.det_transform.augment_keypoints([keypoints_on_image])[0].get_coords_array()

    def augment_keypoints(self, keypoints, shape):
        """
        keypoints: ndarray of 2D keypoints
        shape: shape of the image thes points are placed on
        """
        keypoints_on_images = ia.KeypointsOnImage.from_coords_array(keypoints, shape=shape)
        return self.det_transform.augment_keypoints(keypoints_on_images)

    def augment_segmentation(self, image, num_classes):
        seg = ia.SegmentationMapOnImage(image, shape=image.shape, nb_classes=num_classes)
        aug_seg = self.det_transform.augment_segmentation_maps([seg])[0]
        return aug_seg.get_arr_int()


    def to_deterministic(self):
        # this has to be called before each transform
        self.det_transform = self.transform.to_deterministic()


class TransformWithCrop(ABC):
    """
    Parameters are relative to crop.
    Possibility of cropping augmentation:
        basically only scale should be on crop?
        translation rotation does not matter.

    """
    def __init__(self, affine_transform):
        self.affine_transform = affine_transform

    def update(self, centers, boxes):
        self.affine_transform.update(centers, boxes)


class ImgAugTransformWithCrop(ImgAugTransform, TransformWithCrop):
    def __init__(self, crop_transform, *args, **kwargs):
        ImgAugTransform.__init__(self, *args, **kwargs)
        TransformWithCrop.__init__(self, crop_transform)


class TorchvisionTransform(Transform):
    def __init__(self, transform, normalize, debug, num_copies):
        super().__init__(transform, normalize, debug, num_copies)

    def augment_image(self, image, return_unnormalized=False):
        aug_image = _transform_torchvision(self, image)
        if return_unnormalized:
            aug_image_unnorm = aug_image.copy()

        if not self.debug:
            aug_image = self.normalize(aug_image)

        if return_unnormalized:
            return aug_image, aug_image_unnorm
        else:
            return aug_image

    def augment_segmentation(self, seg, num_classes):
        aug_seg = self.transform(seg)
        return aug_seg

    def augment_images(self, images, return_unnormalized=False):
        raise NotImplementedError()

    def augment_keypoint(self, keypoints, shape):
        raise NotImplementedError()

    def augment_keypoints(self, keypoints, shape):
        raise NotImplementedError()

    def to_deterministic(self):
        # TODO as augment keypoints is not supported
        # this does not matter yet
        pass
