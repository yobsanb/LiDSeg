import glob
import os
import pickle

import numpy as np
import torch
import torch.distributed as dist

from .base_dataset import BaseDataset
from .cityscapes import Cityscapes  # For colormap
from .utils import DatasetDescriptor
from ..transforms import build_transforms
from ..transforms.transforms import apply_paired_transforms


_ACDC_INFORMATION = DatasetDescriptor(
    splits_to_sizes={'train': 1600,  # Approximate, 400 * 4 conditions
                     'val': 406,
                     'test': 2000},
    num_classes=19,
    ignore_label=255,
)

_ACDC_TRAIN_ID_TO_EVAL_ID = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22,
                              23, 24, 25, 26, 27, 28, 31, 32, 33]

_FOLDERS_MAP = {
    'image': 'rgb_anon',
    'label': 'gt',
}

_POSTFIX_MAP = {
    'image': '_rgb_anon',
    'label': '_gt_labelTrainIds',
}

_DATA_FORMAT_MAP = {
    'image': 'png',
    'label': 'png',
}


class ACDC(BaseDataset):
    """ACDC with image/label pairs and an optional paired-clear reference."""

    def __init__(self,
                 root,
                 split,
                 is_train=True,
                 crop_size=(513, 1025),
                 mirror=True,
                 min_scale=0.5,
                 max_scale=2.,
                 scale_step_size=0.25,
                 mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225),
                 img_folder=None,
                 conditions=None,
                 load_clear_reference=False,
                 eval_full_res=True,
                 **kwargs):
        super(ACDC, self).__init__(
            root, split, is_train, crop_size, mirror, min_scale, max_scale,
            scale_step_size, mean, std)

        self.num_classes = _ACDC_INFORMATION.num_classes
        self.ignore_label = _ACDC_INFORMATION.ignore_label
        self.label_pad_value = (self.ignore_label, )
        self.eval_full_res = bool(eval_full_res) and not is_train
        self.conditions = conditions if conditions else ['fog', 'night', 'rain', 'snow']
        self.img_folder = img_folder or _FOLDERS_MAP['image']

        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
        else:
            rank = 0

        if rank == 0:
            self.img_list = self._get_files('image', self.split)
            self.ann_list = self._get_files('label', self.split)
        else:
            self.img_list = None
            self.ann_list = None

        if dist.is_available() and dist.is_initialized():
            self.img_list, self.ann_list = self._broadcast_file_lists(
                (self.img_list, self.ann_list))

        if not self.img_list:
            raise FileNotFoundError(
                "no ACDC images found for "
                f"root={root!r}, conditions={self.conditions!r}, split={split!r}")
        if self.ann_list is not None:
            if not self.ann_list:
                raise FileNotFoundError(
                    "no ACDC labels found for "
                    f"root={root!r}, conditions={self.conditions!r}, split={split!r}")
            ann_dict = {}
            for ann_path in self.ann_list:
                base_name = os.path.splitext(os.path.basename(ann_path))[0]
                base_name = base_name.replace(_POSTFIX_MAP['label'], '')
                if base_name in ann_dict:
                    raise RuntimeError(
                        f"duplicate ACDC annotation stem {base_name!r}")
                ann_dict[base_name] = ann_path
            img_bases = [
                os.path.splitext(os.path.basename(img_path))[0].replace(
                    _POSTFIX_MAP['image'], '')
                for img_path in self.img_list
            ]
            missing = [base for base in img_bases if base not in ann_dict]
            extra = sorted(set(ann_dict).difference(img_bases))
            if missing or extra:
                raise RuntimeError(
                    "ACDC image/label pairing is incomplete for "
                    f"split={split!r}: missing_labels={len(missing)} "
                    f"(e.g. {missing[:2]}), extra_labels={len(extra)} "
                    f"(e.g. {extra[:2]}). Refusing to silently drop samples.")
            self.ann_list = [ann_dict[base] for base in img_bases]

        self.transform = build_transforms(self, is_train)
        self.transform_geo = build_transforms(self, is_train, geo_only=True)
        self.load_clear_reference = bool(load_clear_reference)

    @staticmethod
    def _broadcast_file_lists(value):
        """Broadcast file paths across ranks."""
        if dist.get_backend() != "nccl":
            obj_list = [value]
            dist.broadcast_object_list(obj_list, src=0)
            return obj_list[0]

        device = torch.device("cuda", torch.cuda.current_device())
        if dist.get_rank() == 0:
            payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
            size = len(payload)
        else:
            payload = b""
            size = 0
        size_tensor = torch.tensor([size], dtype=torch.int64, device=device)
        dist.broadcast(size_tensor, src=0)
        size = int(size_tensor.item())
        if dist.get_rank() == 0:
            buffer = torch.frombuffer(
                memoryview(bytearray(payload)), dtype=torch.uint8).to(device)
        else:
            buffer = torch.empty((size,), dtype=torch.uint8, device=device)
        dist.broadcast(buffer, src=0)
        if dist.get_rank() != 0:
            payload = buffer.cpu().numpy().tobytes()
        return pickle.loads(payload)

    def _ref_path_for(self, img_path: str) -> str:
        directory = os.path.dirname(img_path)
        scene = os.path.basename(directory)
        split_dir = os.path.basename(os.path.dirname(directory))
        condition_dir = os.path.dirname(os.path.dirname(directory))
        stem = os.path.splitext(os.path.basename(img_path))[0]
        ref_stem = stem.replace('_rgb_anon', '_rgb_ref_anon')
        return os.path.join(
            condition_dir, split_dir + '_ref', scene, ref_stem + '.png')

    def __getitem__(self, index):
        if not self.load_clear_reference:
            return self._add_condition(index, super().__getitem__(index))
        if self.ann_list is None:
            raise RuntimeError(
                "paired-clear ACDC loading requires labels and is not valid "
                f"for split={self.split!r}")

        img_path = self.img_list[index]
        ann_path = self.ann_list[index]
        image = self.read_image(img_path, 'RGB')
        label = self.read_label(ann_path, self.label_dtype)
        ref_path = self._ref_path_for(img_path)
        if not os.path.isfile(ref_path):
            raise FileNotFoundError(
                "paired ACDC clear reference is required but was not found for "
                f"{img_path}: expected {ref_path}")
        clear = self.read_image(ref_path, 'RGB')
        if image.shape[:2] != clear.shape[:2]:
            raise RuntimeError(
                "paired ACDC image/reference dimensions differ for "
                f"{img_path}: {image.shape[:2]} vs {clear.shape[:2]}")
        image_t, clear_t, label_t = apply_paired_transforms(
            image, clear, label, self.transform, self.transform_geo)
        dataset_dict = {
            'image': image_t,
            'clear_image': clear_t,
            'semantic': torch.as_tensor(label_t.astype('long')),
            'size': np.array(image_t.shape),
            'raw_size': np.array(label.shape),
            'has_paired_clear': True,
        }
        return self._add_condition(index, dataset_dict)

    _CONDITIONS = ('fog', 'night', 'rain', 'snow')
    CONDITION_TO_ID = {condition: i for i, condition in enumerate(_CONDITIONS)}

    def _add_condition(self, index, dataset_dict):
        path = self.img_list[index]
        condition = 'unknown'
        for candidate in self._CONDITIONS:
            if f'/{candidate}/' in path:
                condition = candidate
                break
        dataset_dict['condition'] = condition
        dataset_dict['condition_id'] = self.CONDITION_TO_ID.get(condition, 0)
        dataset_dict['has_paired_clear'] = bool(
            dataset_dict.get('has_paired_clear', False))
        return dataset_dict

    def _get_files(self, data, dataset_split):
        if data == 'label' and dataset_split == 'test':
            return None
        pattern = '*%s.%s' % (_POSTFIX_MAP[data], _DATA_FORMAT_MAP[data])
        folder_name = self.img_folder if data == 'image' else _FOLDERS_MAP[data]
        all_files = []
        for condition in self.conditions:
            search_files = os.path.join(
                self.root, folder_name, condition, dataset_split, '*', pattern)
            all_files.extend(glob.glob(search_files))
        if not all_files:
            for condition in self.conditions:
                search_files = os.path.join(
                    self.root, folder_name, dataset_split, condition, pattern)
                all_files.extend(glob.glob(search_files))
        return sorted(all_files)

    @staticmethod
    def train_id_to_eval_id():
        return _ACDC_TRAIN_ID_TO_EVAL_ID

    @staticmethod
    def create_label_colormap():
        return Cityscapes.create_label_colormap()
