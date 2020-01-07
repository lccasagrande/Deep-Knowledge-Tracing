# Deep-Knowledge-Tracing [![Build Status](https://travis-ci.com/lccasagrande/Deep-Knowledge-Tracing.svg?branch=master)](https://travis-ci.com/lccasagrande/Deep-Knowledge-Tracing) [![Coverage Status](https://coveralls.io/repos/github/lccasagrande/Deep-Knowledge-Tracing/badge.svg?branch=master&kill_cache=1)](https://coveralls.io/github/lccasagrande/Deep-Knowledge-Tracing?branch=master&kill_cache=1)

This repository contains my implementation of [**Deep Knowledge Tracing**](https://github.com/chrispiech/DeepKnowledgeTracing) for the Udacity's Capstone Project.

## Overview

The objective is to predict the probabilities of a student correctly answering a problem not yet seen by him. To this end, we train the model using the [**ASSISTments Skill-builder data 2009-2010**](https://sites.google.com/site/assistmentsdata/home/assistment-2009-2010-data/skill-builder-data-2009-2010) public dataset.

## Requirements

You'll need Python 3.7 x64 and Tensorflow 2.0 to be able to run this project.

If you do not have Python installed yet, it is recommended that you install the [Anaconda](https://www.anaconda.com/download/) distribution of Python, which has almost all packages required in this project. You can also install Python 3.7 x64 from [here](https://www.python.org/downloads/).

Tensorflow 2.0 is installed along with the project. Check the instructions below.

## Instructions

1. Clone the repository and navigate to the downloaded folder.

    ``` bash
    git clone https://github.com/lccasagrande/Deep-Knowledge-Tracing.git
    cd Deep-Knowledge-Tracing
    ```

2. Install required packages:

    - If you want to install with Tensorflow-CPU, type:

        ``` bash
        pip install -e .[tf]
        ```

    - Otherwise, if you want to install with TensorFlow-GPU follow [this guide](https://www.tensorflow.org/install/) to check the necessary NVIDIA software. After that, type:

        ``` bash
        pip install -e .[tf_gpu]
        ```

3. Navigate to the examples folder and:
    - Run the notebook:

        ``` bash
        jupyter notebook DKT.ipynb
        ```

    - Run the python script:

        ``` bash
        python run_dkt.py
        ```

4. [Optional] Analyse the results with tensorboard:

    ``` bash
    tensorboard --logdir=logs
    ```

## Custom Metrics

To implement custom metrics, first decode the label and then calculate the metric. This step is necessary because we encode the skill id with the label to implement the custom loss.

Here's a quick example:

```python
import tensorflow as tf
from deepkt import data_util

def bin_acc(y_true, y_pred):
    true, pred = data_util.get_target(y_true, y_pred)
    return tf.keras.metrics.binary_accuracy(true, pred, threshold=0.5)

dkt_model.compile(optimizer='rmsprop', metrics=[bin_acc])
```

Take a look at [deepkt/metrics.py](https://github.com/lccasagrande/Deep-Knowledge-Tracing/tree/master/deepkt/metrics.py) to check more examples.

## Support

If you have any question or find a bug, please contact me or open an issue. Pull request are also welcome.
