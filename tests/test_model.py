import unittest

import torch

from pytorchyolo.models import load_model
from pytorchyolo.utils.loss import compute_loss
from torch.utils.data import DataLoader
from torch.profiler import profile, record_function, ProfilerActivity
from pytorchyolo.utils.datasets import ListDataset
from pytorchyolo.utils.parse_config import parse_data_config
from pytorchyolo.utils.utils import load_classes
from pytorchyolo.utils.augmentations import AUGMENTATION_TRANSFORMS


class TestModel(unittest.TestCase):
    def setUp(self):
        self.model = load_model("config/yolov3-tiny-voc.cfg", use_dyhead=True)
        self.model.train()
        self.input = torch.randn(16, 3, 416, 416).cuda()

    def test_cpu_time(self):
        with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
            with record_function("model_inference"):
                self.model(self.input)

        prof.export_chrome_trace("tests/tracing/trace_cpu_time.json")
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

    def test_gpu_time(self):
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
            with record_function("model_inference"):
                self.model(self.input)

        prof.export_chrome_trace("tests/tracing/trace_gpu_time.json")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    def test_cpu_memory(self):
        with profile(activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True) as prof:
            self.model(self.input)

        prof.export_chrome_trace("tests/tracing/trace_cpu_memory.json")
        print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))
        print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=10))

    def test_gpu_memory(self):
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, record_shapes=True) as prof:
            self.model(self.input)

        prof.export_chrome_trace("tests/tracing/trace_gpu_memory.json")
        print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))
        print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))


if __name__ == '__main__':
    unittest.main()
