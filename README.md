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

- [DyHead](https://arxiv.org/pdf/2106.08322v1.pdf)
- [YOLOv3](https://arxiv.org/pdf/1804.02767.pdf)

### AWS Environment

We used an EC2 GPU instance with the [AWS Deep Learning AMI (Ubuntu 18.04)](https://aws.amazon.com/marketplace/pp/prodview-x5nivojpquy6y).
However, because we are using a docker environment, the AMI should not matter as long as it is CUDA enabled and contains docker.

### Setup

To run the model, first build your docker image with the following command:

```
docker build .
```

When running locally, use this command to start a container:

```
docker run -dit -p 6006:6006 -v $(PWD):/code --name dyhead dyhead
```

When running on an EC2 server, use this command to start a container:

```
docker run --gpus all -p 6006:6006 -dit -v $(PWD):/code --name dyhead dyhead
```

Once started, you can access your container with:

```
docker exec -it dyhead bash
```

If you need to shutdown your container, to stop it run:

```
docker container stop dyhead
```

To remove it, run:

```
docker container rm dyhead
```

