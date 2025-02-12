"""
Title: Text classification from scratch
Authors: Mark Omernick, Francois Chollet
Date created: 2019/11/06
Last modified: 2020/05/17
Description: Text sentiment classification starting from raw text files.
Accelerator: GPU
"""

"""
## Introduction

This example shows how to do text classification starting from raw text (as
a set of text files on disk). We demonstrate the workflow on the IMDB sentiment
classification dataset (unprocessed version). We use the `TextVectorization` layer for
 word splitting & indexing.
"""

"""
## Setup
"""

import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
import tensorflow as tf
import numpy as np
from keras import layers

model = keras.models.load_model('/workspaces/AI-DEMO/文本处理/text_classification_from_scratch.keras')

dataset = tf.data.Dataset.from_tensor_slices([("The movie was great!",'')])
model.predict(dataset)
