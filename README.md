# Object Detection with Florence-2

## Overview

This project showcases a comprehensive pipeline for object detection using the Florence-2 model. It includes functionalities for object detection inference, dataset handling, model training with Low-Rank Adaptation (LoRA), and automated annotation for both images and videos.

## Features

	1) **Object Detection Inference**: Uses a pre-trained Florence-2 model to perform object detection on images. The results are annotated with bounding boxes and labels.
 
	-Custom Dataset Handling: Defines a JSONLDataset class to handle datasets stored in JSONL format, and a DetectionDataset class for use in PyTorch. This setup enables efficient data loading and preprocessing.
 
	-Model Training with LoRA: Implements a training loop for fine-tuning the Florence-2 model using the LoRA technique. This section includes dataset preparation, model configuration, and training procedures.
 
	-Automated Image Annotation Pipeline: Provides a function to automatically annotate images in a specified folder. The annotations include bounding boxes and labels, which are then saved alongside the images.
 
	-Automated Video Annotation Pipeline: Extracts frames from a video, performs object detection on each frame, and compiles the annotated frames back into a video. This feature allows for comprehensive video analysis.

## Prerequisites

	-Python 3.7+
	-PyTorch
	-Transformers
	-Pillow
	-OpenCV
	-MoviePy
	-TorchVision
	-tqdm

## Installation

Install the required packages using pip:

```pip install torch transformers pillow opencv-python moviepy tqdm```

## Usage

### Object Detection Inference

Load the pre-trained Florence-2 model, process an input image, and perform object detection. The results are annotated and displayed.

### Custom Dataset Handling

Utilize the JSONLDataset and DetectionDataset classes to load and preprocess your dataset. These classes handle JSONL formatted data and integrate seamlessly with PyTorch’s DataLoader.

### Model Training with LoRA

Configure and fine-tune the Florence-2 model using LoRA. The training loop includes dataset preparation, loss calculation, optimization, and evaluation. Checkpoints are saved after each epoch.

### Automated Image Annotation Pipeline

Use the provided function to annotate all images in a specified folder. Annotations include bounding boxes and labels, which are saved alongside the original images.

### Automated Video Annotation Pipeline

Extract frames from a video, perform object detection on each frame, and compile the annotated frames back into a video. This functionality is useful for analyzing and visualizing video data.

### Example

For detailed examples and usage instructions, refer to the example.py file in the repository. This file demonstrates how to perform object detection inference, load custom datasets, train the model, and use the annotation pipelines.

