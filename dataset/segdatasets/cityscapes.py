# Loads Cityscapes semantic dataset. Written by Bowen Cheng (bcheng9@illinois.edu)
#
# Derived from https://github.com/bowenc0221/panoptic-deeplab and used
# under the Apache License 2.0. Copyright Bowen Cheng.
# Modified for LiDSeg.

import glob
import os
import random
import re

from typing import Tuple

import numpy as np
import torch

from .base_dataset import BaseDataset
from .utils import DatasetDescriptor
from ..transforms import build_transforms
from ..transforms.transforms import apply_paired_transforms

CITYSCAPES_IMG_NORM_MEAN_255: Tuple[float, float, float] = (
    123.675,
    116.28,
    103.53,
)
CITYSCAPES_IMG_NORM_STD_255: Tuple[float, float, float] = (
    58.395,
    57.12,
    57.375,
)

CITYSCAPES_IMG_NORM_MEAN: Tuple[float, float, float] = tuple(
    m / 255.0 for m in CITYSCAPES_IMG_NORM_MEAN_255
)
CITYSCAPES_IMG_NORM_STD: Tuple[float, float, float] = tuple(
    s / 255.0 for s in CITYSCAPES_IMG_NORM_STD_255
)

_CITYSCAPES_INFORMATION = DatasetDescriptor(
    splits_to_sizes={'train': 2975,
                     'trainval': 3475,
                     'val': 500,
                     'test': 1525},
    num_classes=19,
    ignore_label=255,
)

_CITYSCAPES_TRAIN_ID_TO_EVAL_ID = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22,
                                   23, 24, 25, 26, 27, 28, 31, 32, 33]

_FOLDERS_MAP = {
    'image': 'leftImg8bit',
    'label': 'gtFine',
}

_POSTFIX_MAP = {
    'image': '_leftImg8bit',
    'label': '_gtFine_labelTrainIds',
}

_DATA_FORMAT_MAP = {
    'image': 'png',
    'label': 'png',
}


class Cityscapes(BaseDataset):
    """Cityscapes semantic segmentation dataset with paired clear image support"""

    def __init__(self,
                 root,
                 split,
                 is_train=True,
                 crop_size=(513, 1025),
                 mirror=True,
                 min_scale=0.5,
                 max_scale=2.,
                 scale_step_size=0.25,
                 mean=CITYSCAPES_IMG_NORM_MEAN,
                 std=CITYSCAPES_IMG_NORM_STD,
                 img_folder=None,
                 load_clear_reference=True,
                 clear_img_folder=os.path.join('foggy_0.00', 'leftImg8bit'),
                 eval_full_res=True,
                 **kwargs):
        super(Cityscapes, self).__init__(
            root, split, is_train, crop_size, mirror, min_scale, max_scale,
            scale_step_size, mean, std)

        self.num_classes = _CITYSCAPES_INFORMATION.num_classes
        self.ignore_label = _CITYSCAPES_INFORMATION.ignore_label
        self.label_pad_value = (self.ignore_label, )
        self.load_clear_reference = load_clear_reference
        self.eval_full_res = bool(eval_full_res) and not is_train

        if img_folder:
            self.img_folder = img_folder
            self.ann_root = root
            if 'Foggy' in img_folder or 'foggy' in img_folder:
                self.clear_img_folder = (
                    os.path.join(root, clear_img_folder)
                    if self.load_clear_reference else None)
        else:
            self.img_folder = _FOLDERS_MAP['image']
            self.ann_root = root

        self.img_list = self._get_files('image', self.split)
        self.ann_list = self._get_files('label', self.split)

        if not self.img_list:
            raise FileNotFoundError(
                "no Cityscapes images found for "
                f"root={root!r}, folder={self.img_folder!r}, split={split!r}")
        if self.ann_list is not None:
            if not self.ann_list:
                raise FileNotFoundError(
                    "no Cityscapes labels found for "
                    f"root={self.ann_root!r}, split={split!r}")
            ann_dict = {}
            for ann_path in self.ann_list:
                base_name = os.path.splitext(os.path.basename(ann_path))[0]
                base_name = base_name.replace('_gtFine_labelTrainIds', '')
                base_name = base_name.replace('_gtFine_labelIds', '')
                if base_name in ann_dict:
                    raise RuntimeError(
                        f"duplicate Cityscapes annotation stem {base_name!r}")
                ann_dict[base_name] = ann_path

            img_bases = []
            for img_path in self.img_list:
                img_base = os.path.splitext(os.path.basename(img_path))[0]
                # Strip the `_foggy_beta_*` and `leftImg8bit` suffixes.
                img_base = re.sub(r'_foggy_beta_[0-9.]+$', '', img_base)
                img_base = img_base.replace('_leftImg8bit', '')
                img_bases.append(img_base)
            missing = [b for b in img_bases if b not in ann_dict]
            extra = sorted(set(ann_dict).difference(img_bases))
            if missing or extra:
                raise RuntimeError(
                    "Cityscapes image/label pairing is incomplete for "
                    f"split={split!r}: missing_labels={len(missing)} "
                    f"(e.g. {missing[:2]}), extra_labels={len(extra)} "
                    f"(e.g. {extra[:2]}). Refusing to silently drop samples.")
            self.ann_list = [ann_dict[b] for b in img_bases]

        self.transform = build_transforms(self, is_train)
        self.transform_geo = build_transforms(self, is_train, geo_only=True)

    _FOG_BETA_TO_NAME = (
        ("foggy_0.005", "light_fog", 1),
        ("foggy_0.01",  "medium_fog", 2),
        ("foggy_0.02",  "dense_fog",  3),
    )

    def _extract_condition(self, img_path: str) -> str:
        p = img_path.lower()
        for key, name, _ in self._FOG_BETA_TO_NAME:
            if key in p:
                return name
        return "clean"

    def _extract_condition_id(self, img_path: str) -> int:
        p = img_path.lower()
        for key, _, cid in self._FOG_BETA_TO_NAME:
            if key in p:
                return cid
        return 0

    def __getitem__(self, index):
        """Foggy + paired clear with identical geometric transforms (train or val)."""
        use_paired_clear = (
            self.load_clear_reference
            and getattr(self, 'clear_img_folder', None) is not None
        )
        if not use_paired_clear:
            dataset_dict = super().__getitem__(index)
            dataset_dict['condition'] = self._extract_condition(
                self.img_list[index])
            dataset_dict['condition_id'] = self._extract_condition_id(
                self.img_list[index])
            dataset_dict['has_paired_clear'] = False
            return dataset_dict

        foggy_img_path = self.img_list[index]
        img_basename = os.path.basename(foggy_img_path)
        city_name = os.path.basename(os.path.dirname(foggy_img_path))
        clear_candidates = self._clear_reference_candidates(
            city_name, img_basename)
        clear_img_path = next(
            (p for p in clear_candidates if os.path.isfile(p)), None)
        if clear_img_path is None:
            raise FileNotFoundError(
                "paired clear reference is required but was not found for "
                f"{foggy_img_path}. Looked for: {clear_candidates}")

        image = self.read_image(foggy_img_path, 'RGB')
        clear = self.read_image(clear_img_path, 'RGB')
        label = self.read_label(self.ann_list[index], self.label_dtype)

        if image.shape[:2] != clear.shape[:2]:
            raise RuntimeError(
                "paired Cityscapes image/reference dimensions differ for "
                f"{foggy_img_path}: {image.shape[:2]} vs {clear.shape[:2]}")
        if self.pre_augmentation_transform is not None:
            # Run the pre-transform hook with paired geometry locked.
            rs_py = random.getstate()
            rs_np = np.random.get_state()
            # Each branch receives the same raw label.
            raw_label = label.copy()
            image, image_label = self.pre_augmentation_transform(
                image, raw_label.copy())
            after_py = random.getstate()
            after_np = np.random.get_state()
            try:
                random.setstate(rs_py)
                np.random.set_state(rs_np)
                clear, clear_label = self.pre_augmentation_transform(
                    clear, raw_label.copy())
            finally:
                random.setstate(after_py)
                np.random.set_state(after_np)
            if image_label.shape[:2] != clear_label.shape[:2]:
                raise RuntimeError(
                    "paired Cityscapes pre-transform produced different label "
                    f"geometry: adverse={image_label.shape[:2]}, "
                    f"clear={clear_label.shape[:2]}")
            if not np.array_equal(image_label, clear_label):
                raise RuntimeError(
                    "paired Cityscapes pre-transform did not apply identical "
                    "geometry to adverse and clear labels")
            label = image_label

        image_t, clear_t, label_t = apply_paired_transforms(
            image, clear, label, self.transform, self.transform_geo)

        return {
            'image': image_t,
            'clear_image': clear_t,
            'semantic': torch.as_tensor(label_t.astype('long')),
            'size': np.array(image_t.shape),
            'raw_size': np.array(label.shape),
            'condition': self._extract_condition(foggy_img_path),
            'condition_id': self._extract_condition_id(foggy_img_path),
            'has_paired_clear': True,
        }

    def _clear_reference_candidates(self, city_name: str,
                                    foggy_basename: str):
        """Return supported clear-reference spellings for one foggy image."""
        stem, ext = os.path.splitext(foggy_basename)
        canonical = re.sub(r'_foggy_beta_[0-9.]+$', '', stem) + ext
        zero_fog = re.sub(
            r'_foggy_beta_[0-9.]+$', '_foggy_beta_0.00', stem) + ext
        base = os.path.join(self.clear_img_folder, self.split, city_name)
        candidates = [
            os.path.join(base, canonical),
            os.path.join(base, zero_fog),
            os.path.join(base, foggy_basename),
        ]
        # Deduplicate, preserving order.
        return list(dict.fromkeys(candidates))

    def _get_files(self, data, dataset_split):
        if data == 'label' and dataset_split == 'test':
            return None
        pattern = '*%s.%s' % (_POSTFIX_MAP[data], _DATA_FORMAT_MAP[data])

        if data == 'image' and hasattr(self, 'img_folder'):
            folder_name = self.img_folder
            root_path = self.root
        elif data == 'label' and hasattr(self, 'ann_root'):
            folder_name = _FOLDERS_MAP[data]
            root_path = self.ann_root
        else:
            folder_name = _FOLDERS_MAP[data]
            root_path = self.root

        # Foggy Cityscapes appends `_foggy_beta_<beta>` after the leftImg8bit suffix.
        if data == 'image' and ('Foggy' in folder_name or 'foggy' in folder_name):
            pattern = '*_leftImg8bit*.%s' % _DATA_FORMAT_MAP[data]

        search_files = os.path.join(
            root_path, folder_name, dataset_split, '*', pattern)
        filenames = glob.glob(search_files)
        return sorted(filenames)

    @staticmethod
    def train_id_to_eval_id():
        return _CITYSCAPES_TRAIN_ID_TO_EVAL_ID

    def _convert_train_id_to_eval_id(self, prediction):
        converted_prediction = prediction.copy()
        for train_id, eval_id in enumerate(self.train_id_to_eval_id()):
            converted_prediction[prediction == train_id] = eval_id
        return converted_prediction

    @staticmethod
    def create_label_colormap():
        colormap = np.zeros((256, 3), dtype=np.uint8)
        colormap[0] = [128, 64, 128]
        colormap[1] = [244, 35, 232]
        colormap[2] = [70, 70, 70]
        colormap[3] = [102, 102, 156]
        colormap[4] = [190, 153, 153]
        colormap[5] = [153, 153, 153]
        colormap[6] = [250, 170, 30]
        colormap[7] = [220, 220, 0]
        colormap[8] = [107, 142, 35]
        colormap[9] = [152, 251, 152]
        colormap[10] = [70, 130, 180]
        colormap[11] = [220, 20, 60]
        colormap[12] = [255, 0, 0]
        colormap[13] = [0, 0, 142]
        colormap[14] = [0, 0, 70]
        colormap[15] = [0, 60, 100]
        colormap[16] = [0, 80, 100]
        colormap[17] = [0, 0, 230]
        colormap[18] = [119, 11, 32]
        return colormap
