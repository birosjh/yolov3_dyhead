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
        self.dyhead_num_convs = [0, 4, 5, 6, 7]
        for layer_num in self.dyhead_num_convs:
            if not layer_num:
                model = load_model("config/yolov3-tiny-voc.cfg".format(layer_num), use_dyhead=False)
            else:
                model = load_model("config/yolov3-tiny-voc-{}layers.cfg".format(layer_num), use_dyhead=True)
            model.train()
            self.models.append(model)
        self.input = torch.randn(16, 3, 416, 416).cuda()

    def test_model_network_structure(self):
        for model_info in zip(self.models, self.dyhead_num_convs):
            model = model_info[0]
            layer_num = model_info[1]
            if not layer_num:
                print("\nModel - YT ======>")
            else:
                print("\nModel - YTD-{}-LAYERS ======>".format(layer_num))
            print(model)

    def test_model_output_shape(self):
        for model_info in zip(self.models, self.dyhead_num_convs):
            model = model_info[0]
            layer_num = model_info[1]
            outputs = model(self.input)
            if not layer_num:
                print("\nModel - YT ======>")
            else:
                print("\nModel - YTD-{}-LAYERS ======>\n".format(layer_num))
            print("Input: (16, 3, 416, 416), Output length: ", len(outputs))
            for output in outputs:
                print("Output shape: ", output.shape)

    def test_cpu_time(self):
        for model_info in zip(self.models, self.dyhead_num_convs):
            model = model_info[0]
            layer_num = model_info[1]
            with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
                with record_function("model_inference"):
                    model(self.input)

            # prof.export_chrome_trace("tests/tracing/trace_cpu_time_{}Layers.json".format(layer_num))
            if not layer_num:
                print("\nModel - YT ======>")
            else:
                print("\nModel - YTD-{}-LAYERS ======>".format(layer_num))
            print("\nCPU TIME TOTAL")
            print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

    def test_gpu_time(self):
        for model_info in zip(self.models, self.dyhead_num_convs):
            model = model_info[0]
            layer_num = model_info[1]
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
                with record_function("model_inference"):
                    model(self.input)

            # prof.export_chrome_trace("tests/tracing/trace_gpu_time_{}Layers.json".format(layer_num))
            if not layer_num:
                print("\nModel - YT ======>")
            else:
                print("\nModel - YTD-{}-LAYERS ======>".format(layer_num))
            print("\nCUDA TIME TOTAL")
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    def test_cpu_memory(self):
        for model_info in zip(self.models, self.dyhead_num_convs):
            model = model_info[0]
            layer_num = model_info[1]
            with profile(activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True) as prof:
                model(self.input)

            # prof.export_chrome_trace("tests/tracing/trace_cpu_memory_{}Layers.json".format(layer_num))
            if not layer_num:
                print("\nModel - YT ======>")
            else:
                print("\nModel - YTD-{}-LAYERS ======>".format(layer_num))
            print("\nSELF CPU MEMORY USAGE")
            print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))

            print("\nCPU MEMORY USAGE")
            print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=10))

    def test_gpu_memory(self):
        for model_info in zip(self.models, self.dyhead_num_convs):
            model = model_info[0]
            layer_num = model_info[1]
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, record_shapes=True) as prof:
                model(self.input)

            # prof.export_chrome_trace("tests/tracing/trace_gpu_memory_{}Layers.json".format(layer_num))
            if not layer_num:
                print("\nModel - YT ======>")
            else:
                print("\nModel - YTD-{}-LAYERS ======>".format(layer_num))
            print("\nSELF CUDA MEMORY USAGE")
            print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))

            print("\nCUDA MEMORY USAGE")
            print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))

    def test_num_params(self):
        for model_info in zip(self.models, self.dyhead_num_convs):
            model = model_info[0]
            layer_num = model_info[1]
            total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            learnable_params = sum(p.numel() for p in model.parameters())

            if not layer_num:
                print("\nModel - YT ======>")
            else:
                print("\nYTD-{}-LAYERS ======>".format(layer_num))
            print("Total Parameters: ", total_params)
            print("Learnable Parameters: ", learnable_params)
            print("\n")


if __name__ == '__main__':
    unittest.main()
