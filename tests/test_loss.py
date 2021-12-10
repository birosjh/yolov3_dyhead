import unittest

import torch

from pytorchyolo.models import load_model
from pytorchyolo.utils.loss import compute_loss
from torch.utils.data import DataLoader
from pytorchyolo.utils.datasets import ListDataset
from pytorchyolo.utils.parse_config import parse_data_config
from pytorchyolo.utils.utils import load_classes
from pytorchyolo.utils.augmentations import AUGMENTATION_TRANSFORMS


class TestLoss(unittest.TestCase):

    def setUp(self):
        self.model = load_model("config/yolov3-voc.cfg")

        self.model.train()

        # Get data configuration
        data_config = parse_data_config("config/voc.data")
        train_path = data_config["train"]
        #valid_path = data_config["valid"]
        #class_names = load_classes(data_config["names"])
        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batch_size = self.model.hyperparams['batch'] // self.model.hyperparams['subdivisions']
        img_size = self.model.hyperparams['height']

        dataset = ListDataset(
            train_path,
            img_size=img_size,
            multiscale=True,
            transform=AUGMENTATION_TRANSFORMS)
        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            collate_fn=dataset.collate_fn,
            worker_init_fn=-1)


    def test_loss(self):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        _, images, targets = next(iter(self.dataloader))

        images = images.to(device, non_blocking=True)   
        targets = targets.to(device)
        
        outputs = self.model(images)

        print(outputs.shape)

        loss, loss_components = compute_loss(outputs, targets, self.model)



if __name__ == '__main__':
    unittest.main()