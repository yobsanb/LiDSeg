# Segmentation transforms. Written by Bowen Cheng (bcheng9@illinois.edu).
#
# Derived from https://github.com/bowenc0221/panoptic-deeplab and used
# under the Apache License 2.0. Copyright Bowen Cheng.
# Modified for LiDSeg.

import math
import random
import warnings

import cv2
import numpy as np
from torchvision.transforms import functional as F


class Compose(object):
    """Composes a sequence of transforms."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, label):
        for transform in self.transforms:
            image, label = transform(image, label)
        return image, label

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for transform in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(transform)
        format_string += "\n)"
        return format_string


def apply_paired_transforms(image, clear, label, image_transform,
                            clear_geometry_transform):
    """Apply matched geometry to an adverse/clear pair, restoring RNG state."""
    if image.shape[:2] != clear.shape[:2]:
        raise ValueError(
            "paired adverse and clear images must have identical spatial "
            f"dimensions, got {image.shape[:2]} and {clear.shape[:2]}")
    if label is None:
        raise ValueError("paired transforms require a semantic label")
    if label.shape[:2] != image.shape[:2]:
        raise ValueError(
            "paired image/label geometry mismatch: "
            f"image={image.shape[:2]}, label={label.shape[:2]}")

    # Give each branch an independent copy of the raw label.
    raw_label = label.copy()
    before_py = random.getstate()
    before_np = np.random.get_state()
    image_t, label_t = image_transform(image, raw_label.copy())
    after_py = random.getstate()
    after_np = np.random.get_state()
    try:
        random.setstate(before_py)
        np.random.set_state(before_np)
        clear_t, clear_label_t = clear_geometry_transform(clear, raw_label.copy())
    finally:
        random.setstate(after_py)
        np.random.set_state(after_np)

    if image_t.shape[-2:] != clear_t.shape[-2:]:
        raise RuntimeError(
            "paired transform produced different tensor geometry: "
            f"adverse={tuple(image_t.shape[-2:])}, "
            f"clear={tuple(clear_t.shape[-2:])}")
    if label_t.shape[-2:] != clear_label_t.shape[-2:]:
        raise RuntimeError(
            "paired transform produced different label geometry: "
            f"adverse={tuple(label_t.shape[-2:])}, "
            f"clear={tuple(clear_label_t.shape[-2:])}")
    if not np.array_equal(label_t, clear_label_t):
        raise RuntimeError(
            "paired transform did not apply identical geometry to adverse and "
            "clear labels")
    return image_t, clear_t, label_t


class ToTensor(object):
    """Converts image to torch Tensor."""

    def __call__(self, image, label):
        return F.to_tensor(image), label


class Normalize(object):
    """Normalizes image by mean and std."""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, label):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, label


class RandomScale(object):
    """Scale image and label by a random factor."""

    def __init__(self, min_scale, max_scale, scale_step_size):
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.scale_step_size = scale_step_size

    @staticmethod
    def get_random_scale(min_scale_factor, max_scale_factor, step_size):
        """Sample a scale factor in [min_scale, max_scale]."""
        if min_scale_factor < 0 or min_scale_factor > max_scale_factor:
            raise ValueError('Unexpected value of min_scale_factor.')
        if min_scale_factor == max_scale_factor:
            return min_scale_factor
        if step_size == 0:
            return random.uniform(min_scale_factor, max_scale_factor)
        num_steps = int((max_scale_factor - min_scale_factor) / step_size + 1)
        scale_factors = np.linspace(min_scale_factor, max_scale_factor, num_steps)
        np.random.shuffle(scale_factors)
        return scale_factors[0]

    def __call__(self, image, label):
        f_scale = self.get_random_scale(
            self.min_scale, self.max_scale, self.scale_step_size)
        image_dtype = image.dtype
        label_dtype = label.dtype
        image = cv2.resize(image.astype(np.float32), None, fx=f_scale, fy=f_scale,
                           interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label.astype(np.float32), None, fx=f_scale, fy=f_scale,
                           interpolation=cv2.INTER_NEAREST)
        return image.astype(image_dtype), label.astype(label_dtype)


class RandomCrop(object):
    """Crop with mean-image and ignore-label padding."""

    def __init__(self, crop_h, crop_w, pad_value, ignore_label, random_pad,
                 random_crop=True):
        self.crop_h = crop_h
        self.crop_w = crop_w
        self.pad_value = pad_value
        self.ignore_label = ignore_label
        self.random_pad = random_pad
        self.random_crop = random_crop

    def __call__(self, image, label):
        img_h, img_w = image.shape[0], image.shape[1]
        image_dtype = image.dtype
        label_dtype = label.dtype
        pad_h = max(self.crop_h - img_h, 0)
        pad_w = max(self.crop_w - img_w, 0)
        if pad_h > 0 or pad_w > 0:
            if self.random_pad:
                pad_top = random.randint(0, pad_h)
                pad_bottom = pad_h - pad_top
                pad_left = random.randint(0, pad_w)
                pad_right = pad_w - pad_left
            else:
                pad_top, pad_bottom, pad_left, pad_right = 0, pad_h, 0, pad_w
            img_pad = cv2.copyMakeBorder(
                image, pad_top, pad_bottom, pad_left, pad_right,
                cv2.BORDER_CONSTANT, value=self.pad_value)
            label_pad = cv2.copyMakeBorder(
                label, pad_top, pad_bottom, pad_left, pad_right,
                cv2.BORDER_CONSTANT, value=self.ignore_label)
        else:
            img_pad, label_pad = image, label
        img_h, img_w = img_pad.shape[0], img_pad.shape[1]
        if self.random_crop:
            h_off = random.randint(0, img_h - self.crop_h)
            w_off = random.randint(0, img_w - self.crop_w)
        else:
            h_off = (img_h - self.crop_h) // 2
            w_off = (img_w - self.crop_w) // 2
        image = np.asarray(
            img_pad[h_off:h_off + self.crop_h, w_off:w_off + self.crop_w],
            np.float32)
        label = np.asarray(
            label_pad[h_off:h_off + self.crop_h, w_off:w_off + self.crop_w],
            np.float32)
        return image.astype(image_dtype), label.astype(label_dtype)


class RandomHorizontalFlip(object):
    """Applies random flip augmentation."""

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, label):
        if random.random() < self.prob:
            image = image[:, ::-1].copy()
            label = label[:, ::-1].copy()
        return image, label


class ColorJitter(object):
    """Random brightness, contrast and saturation jitter."""

    def __init__(self, brightness=0, contrast=0, saturation=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation

    def __call__(self, image, label):
        if self.brightness > 0 or self.contrast > 0 or self.saturation > 0:
            from PIL import Image
            pil_image = Image.fromarray(image.astype('uint8'))
            if random.random() < 0.8:
                if self.brightness > 0:
                    pil_image = F.adjust_brightness(
                        pil_image,
                        random.uniform(max(0, 1 - self.brightness),
                                       1 + self.brightness))
                if self.contrast > 0:
                    pil_image = F.adjust_contrast(
                        pil_image,
                        random.uniform(max(0, 1 - self.contrast),
                                       1 + self.contrast))
                if self.saturation > 0:
                    pil_image = F.adjust_saturation(
                        pil_image,
                        random.uniform(max(0, 1 - self.saturation),
                                       1 + self.saturation))
            image = np.array(pil_image)
        return image, label


class GaussianBlur(object):
    """Random Gaussian blur."""

    def __init__(self, prob=0.5, kernel_size=(5, 5), sigma=(0.1, 2.0)):
        self.prob = prob
        self.kernel_size = kernel_size
        self.sigma = sigma

    def __call__(self, image, label):
        if random.random() < self.prob:
            sigma_val = random.uniform(self.sigma[0], self.sigma[1])
            image = cv2.GaussianBlur(
                image, self.kernel_size, sigmaX=sigma_val, sigmaY=sigma_val)
        return image, label


class Resize(object):
    """Resize image and label to a fixed size."""

    def __init__(self, min_resize_value=None, max_resize_value=None,
                 resize_factor=None, keep_aspect_ratio=True,
                 align_corners=False):
        if min_resize_value is not None and min_resize_value < 0:
            min_resize_value = None
        if max_resize_value is not None and max_resize_value < 0:
            max_resize_value = None
        if resize_factor is not None and resize_factor < 0:
            resize_factor = None
        self.min_resize_value = min_resize_value
        self.max_resize_value = max_resize_value
        self.resize_factor = resize_factor
        self.keep_aspect_ratio = keep_aspect_ratio
        self.align_corners = align_corners
        if self.align_corners:
            warnings.warn('`align_corners = True` is not supported by opencv.')
        if self.max_resize_value is not None and self.resize_factor is not None:
            self.max_resize_value = (
                self.max_resize_value
                - (self.max_resize_value - 1) % self.resize_factor)

    def __call__(self, image, label):
        if self.min_resize_value is None:
            return image, label
        orig_height, orig_width, _ = image.shape
        orig_min_size = np.minimum(orig_height, orig_width)
        large_scale_factor = self.min_resize_value / orig_min_size
        large_height = int(math.floor(orig_height * large_scale_factor))
        large_width = int(math.floor(orig_width * large_scale_factor))
        large_size = np.array([large_height, large_width])
        new_size = large_size
        if self.max_resize_value is not None:
            orig_max_size = np.maximum(orig_height, orig_width)
            small_scale_factor = self.max_resize_value / orig_max_size
            small_height = int(math.floor(orig_height * small_scale_factor))
            small_width = int(math.floor(orig_width * small_scale_factor))
            small_size = np.array([small_height, small_width])
            if np.max(large_size) > self.max_resize_value:
                new_size = small_size
        if self.resize_factor is not None:
            new_size += (self.resize_factor - (new_size - 1) % self.resize_factor) % self.resize_factor
            if self.max_resize_value is not None:
                new_size[new_size > self.max_resize_value] -= self.resize_factor
        if not self.keep_aspect_ratio:
            new_size = [np.max(new_size), np.max(new_size)]
        image_dtype = image.dtype
        label_dtype = label.dtype
        image = cv2.resize(image.astype(np.float32), (new_size[1], new_size[0]),
                           interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label.astype(np.float32), (new_size[1], new_size[0]),
                           interpolation=cv2.INTER_NEAREST)
        return image.astype(image_dtype), label.astype(label_dtype)


def build_transforms(dataset, is_train=True, geo_only=False):
    """Build deterministic native validation or stochastic training transforms."""
    mean = dataset.mean
    std = dataset.std
    to_tensor = [ToTensor(), Normalize(mean, std)]
    if not is_train and getattr(dataset, "eval_full_res", False):
        return Compose(to_tensor)

    if is_train:
        min_scale = dataset.min_scale
        max_scale = dataset.max_scale
        scale_step_size = dataset.scale_step_size
        flip_prob = 0.5 if dataset.mirror else 0
    else:
        min_scale = max_scale = 1
        scale_step_size = 0
        flip_prob = 0
    transform_list = [
        RandomScale(min_scale, max_scale, scale_step_size),
        RandomCrop(dataset.crop_h, dataset.crop_w, dataset.pad_value,
                   dataset.label_pad_value, random_pad=is_train,
                   random_crop=is_train),
        RandomHorizontalFlip(flip_prob),
    ]
    if is_train and not geo_only:
        transform_list.extend([
            ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            GaussianBlur(prob=0.5, kernel_size=(5, 5), sigma=(0.1, 2.0)),
        ])
    transform_list.extend(to_tensor)
    return Compose(transform_list)
