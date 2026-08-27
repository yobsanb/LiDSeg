# From https://github.com/tensorflow/models research/deeplab/datasets/data_generator.py
#
# Used under the Apache License 2.0.
# Copyright The TensorFlow Authors.

import collections

DatasetDescriptor = collections.namedtuple(
    'DatasetDescriptor',
    [
        'splits_to_sizes',  # Splits of the dataset into training, val and test.
        'num_classes',  # Number of semantic classes, including the
        'ignore_label',  # Ignore label value.
    ])
