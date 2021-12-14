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
        self.models = []
        for num_layers in [4, 5, 6, 7]:
            model = load_model("config/yolov3-tiny-voc-{}layers.cfg".format(num_layers), use_dyhead=True)
            model.train()
            self.models.append(model)
        self.input = torch.randn(16, 3, 416, 416).cuda()

    def test_model_network_structure(self):
        for i, model in enumerate(self.models):
            print("Model - YTD-{}layers\n".format(i+4))
            print(model)

    def test_model_output_size(self):
        for i, model in enumerate(self.models):
            outputs = model(self.input)
            print("Model - YTD-{}layers output shape\n".format(i + 4))
            print("Input: (16, 3, 416, 416), output length: ", len(outputs))
            for output in outputs:
                print("Output shape: ", outputs.shape)

    def test_cpu_time(self):
        for i, model in enumerate(self.models):
            with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
                with record_function("model_inference"):
                    model(self.input)

            # prof.export_chrome_trace("tests/tracing/trace_cpu_time_{}Layers.json".format(i+4))
            print("\nCPU TIME TOTAL DYHEAD {} LAYERS ======>".format(i+4))
            print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

    def test_gpu_time(self):
        for i, model in enumerate(self.models):
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
                with record_function("model_inference"):
                    model(self.input)

            # prof.export_chrome_trace("tests/tracing/trace_gpu_time_{}Layers.json".format(i+4))
            print("\nCUDA TIME TOTAL DYHEAD {} LAYERS ======>".format(i+4))
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    def test_cpu_memory(self):
        for i, model in enumerate(self.models):
            with profile(activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True) as prof:
                model(self.input)

            # prof.export_chrome_trace("tests/tracing/trace_cpu_memory_{}Layers.json".format(i+4))
            print("\nSELF CPU MEMORY USAGE DYHEAD {} LAYERS ======>".format(i+4))
            print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))
            print("\nCPU MEMORY USAGE DYHEAD {} LAYERS ======>".format(i+4))
            print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=10))

    def test_gpu_memory(self):
        for i, model in enumerate(self.models):
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, record_shapes=True) as prof:
                model(self.input)

            # prof.export_chrome_trace("tests/tracing/trace_gpu_memory_{}Layers.json".format(i+4))
            print("\nSELF CUDA MEMORY USAGE DYHEAD {} LAYERS ======>".format(i+4))
            print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))
            print("\nCUDA MEMORY USAGE DYHEAD {} LAYERS ======>".format(i+4))
            print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))


if __name__ == '__main__':
    unittest.main()
