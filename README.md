## Overview

This repository contains my implementation of [**Deep Knowledge Tracing**](https://github.com/chrispiech/DeepKnowledgeTracing) for Udacity's Capstone Project. 

## Objective

Build and train a **LSTM network** to predict the probabilities of a student answering correctly a problem not yet seen by him using the [**ASSISTments Skill-builder data 2009-2010**](https://sites.google.com/site/assistmentsdata/home/assistment-2009-2010-data/skill-builder-data-2009-2010) public dataset.

## Results
This is the best results obtained by comparing the validation loss between each network configuration attempted.

| Test Data (%) | AUC |
| --- | --- |
| 20% | 0,85 |

The results, configuration and model's weights of each attempt can be found in the "Log" folder. 

## Requirements

You'll need Python 3.x x64 to be able to run theses projects. 

If you do not have Python installed yet, it is recommended that you install the [Anaconda](https://www.anaconda.com/download/) distribution of Python, which has almost all packages required in these projects. 

You can also install Python 3.x x64 from [here](https://www.python.org/downloads/)

## Instructions

1. Clone the repository and navigate to the downloaded folder.
```	
git clone https://github.com/lccasagrande/Deep-Knowledge-Tracing.git
cd Deep-Knowledge-Tracing
```

2. Install required packages:
	- If you already has TensorFlow installed, type:  
	```
	pip install -e .
	```
	- If you want to install with TensorFlow-GPU, follow [this guide](https://www.tensorflow.org/install/) to check the necessary NVIDIA software on your system. After that, type:
	```
	pip install -e .[tf_gpu]
	```
	- If you want to install with Tensorflow-CPU, type:
	```
	pip install -e .[tf]
	```

3. Navigate to the src folder and open the notebook.
```	
cd src
jupyter notebook DKT.ipynb
```

4. The most important step: Have fun !!!

If you have any questions or find a bug, please contact me!