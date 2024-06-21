# Object_detection_(CAR)
 The notebook car_object_detection.ipynb contains 67 cells, with a mix of markdown and code cells. The metadata indicates it was designed for use with Google Colab and a GPU accelerator. The code is referenced from coursera tutorial and modified for car as object and it's detection.

Specific TensorFlow Checkpoint for SSD ResNet 50 Version 1 Model
**Overview**
The SSD ResNet 50 v1 FPN model is a state-of-the-art object detection model that balances accuracy and efficiency. The model combines the strengths of the SSD architecture, which provides good detection speed, with the ResNet-50 backbone, known for its deep feature extraction capabilities. The model also includes a Feature Pyramid Network (FPN) to improve detection accuracy at multiple scales.

**Key Feature**s
**Architecture**: SSD (Single Shot MultiBox Detector)
**Backbone**: ResNet-50
**Input Resolution**: 640x640 pixels
**Pre-trained Dataset**: COCO (Common Objects in Context)
**Feature Pyramid Network (FPN)**: Enhances the model's ability to detect objects at different scales.
**Checkpoint File**: The checkpoint file contains the pre-trained weights for the model, which can be fine-tuned on a custom dataset.

Usage

**Downloading the Checkpoint:
**The model checkpoint can be downloaded from the TensorFlow Model Zoo. The following command downloads the checkpoint file:

!wget http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz

Extracting the Checkpoint:
The downloaded file is a tar.gz archive. It needs to be decompressed to access the checkpoint files.

!tar -xf ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz

Moving the Checkpoint:
For convenience, the checkpoint files can be moved to a specific directory, such as the test_data folder within the TensorFlow models research directory.

!mv ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/checkpoint models/research/object_detection/test_data/

Integration with TensorFlow Object Detection API
The TensorFlow Object Detection API provides utilities to use the pre-trained SSD ResNet 50 v1 FPN model. 

Here is a brief guide on how to integrate this model:
**Import Required Libraries:
**Ensure that the necessary modules from the TensorFlow Object Detection API are imported.

python

_from object_detection.builders import model_builder
from object_detection.utils import config_util
Load the Pipeline Configuration:
Load the configuration file specific to the SSD ResNet 50 v1 FPN model. This file defines the model architecture and other training parameters._

python

_configs = config_util.get_configs_from_pipeline_file('models/research/object_detection/configs/tf2/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.config')
model_config = configs['model']_

Build the Detection Model:
Use the configuration to build the detection model.

python
_detection_model = model_builder.build(model_config=model_config, is_training=False)_

Restore the Checkpoint:
Restore the model weights from the checkpoint.

python
_ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore('models/research/object_detection/test_data/checkpoint/ckpt-0').expect_partial()
_
