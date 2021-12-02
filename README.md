# Final Project: DyHead YOLOv3

Members:
-	Hoyean Hwang
-	Waris Kulnguan
-	Jun Xiong Tan
-	Jacob H. Biros

This project is a direct copy of the [PyTorch-YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3) project with the DyHead Adapter implemented directly inside the codebase.

It references the code from the [DyHead](https://github.com/microsoft/DynamicHead/tree/master/dyhead) project to some extent.

### Outline

This is an implementation of the DyHead adapter for the YOLOv3 object detection model.  Our goal is to benchmark the DyHead adapted version along with the original YOLOv3 to see how performance increases, and determine if there are any tradeoffs for any gains obtained.

For benchmarking we will be using the [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/) dataset for training and validation.

Relevant Research Papers:

[DyHead](https://arxiv.org/pdf/2106.08322v1.pdf)
[YOLOv3](https://arxiv.org/pdf/1804.02767.pdf)
