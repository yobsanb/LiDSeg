import os
import glob
import argparse
from PIL import Image
import numpy as np
from tqdm import tqdm

LABEL_MAPPING = {
    0: 255,   # unlabeled
    1: 255,   # ego vehicle
    2: 255,   # rectification border
    3: 255,   # out of roi
    4: 255,   # static
    5: 255,   # dynamic
    6: 255,   # ground
    7: 0,     # road
    8: 1,     # sidewalk
    9: 255,   # parking
    10: 255,  # rail track
    11: 2,    # building
    12: 3,    # wall
    13: 4,    # fence
    14: 255,  # guard rail
    15: 255,  # bridge
    16: 255,  # tunnel
    17: 5,    # pole
    18: 255,  # polegroup
    19: 6,    # traffic light
    20: 7,    # traffic sign
    21: 8,    # vegetation
    22: 9,    # terrain
    23: 10,   # sky
    24: 11,   # person
    25: 12,   # rider
    26: 13,   # car
    27: 14,   # truck
    28: 15,   # bus
    29: 255,  # caravan
    30: 255,  # trailer
    31: 16,   # train
    32: 17,   # motorcycle
    33: 18,   # bicycle
    -1: 255,  # license plate
}


def convert_label(label_path, output_path):
    """Convert a single label file from labelIds to labelTrainIds"""
    label = np.array(Image.open(label_path))

    train_id_label = np.zeros_like(label, dtype=np.uint8)
    train_id_label.fill(255)  # Default to ignore

    for label_id, train_id in LABEL_MAPPING.items():
        train_id_label[label == label_id] = train_id

    Image.fromarray(train_id_label).save(output_path)


def main():
    parser = argparse.ArgumentParser(description='Convert ACDC labels')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to ACDC dataset root')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing labelTrainIds files')
    args = parser.parse_args()

    pattern = os.path.join(args.input, 'gt', '*', '*', '*', '*_gt_labelIds.png')
    label_files = glob.glob(pattern)

    if not label_files:
        pattern = os.path.join(args.input, 'gt', '*', '*', '*_gt_labelIds.png')
        label_files = glob.glob(pattern)

    if not label_files:
        print(f"No labelIds files found in {args.input}")
        print("Expected pattern: gt/<condition>/<split>/<city>/*_gt_labelIds.png")
        return

    print(f"Found {len(label_files)} label files to convert")

    converted = 0
    skipped = 0

    for label_path in tqdm(label_files, desc='Converting'):
        output_path = label_path.replace('_gt_labelIds.png', '_gt_labelTrainIds.png')

        if os.path.exists(output_path) and not args.overwrite:
            skipped += 1
            continue

        convert_label(label_path, output_path)
        converted += 1

    print(f"\nConversion complete:")
    print(f"  Converted: {converted}")
    print(f"  Skipped (already exists): {skipped}")


if __name__ == '__main__':
    main()
