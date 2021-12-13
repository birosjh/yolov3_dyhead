import unittest

import torch

from pytorchyolo.models import load_model
from pytorchyolo.utils.loss import compute_loss
from torch.utils.data import DataLoader
from pytorchyolo.utils.datasets import ListDataset
from pytorchyolo.utils.parse_config import parse_data_config
from pytorchyolo.utils.utils import load_classes
from pytorchyolo.utils.augmentations import AUGMENTATION_TRANSFORMS


class TestModel(unittest.TestCase):

    def test_model(self):

        model = load_model("config/yolov3-voc.cfg", use_dyhead=True)
        model.train()

        print(model)