## Project Overview

This repository contains my implementation of **Deep Knowledge Tracing** for Udacity's Capstone Project. 

The objective was to build and train a LSTM network to predict the probabilities of a student answering correctly a problem not yet seen by him.

To train the model it was used the public dataset called "**ASSISTments Skill-builder data 2009-2010**".

## Project Results
To evaluate the performance of the model, 20% of the dataset was separated for testing and another 20% for validation at the end of each epoch. 

After some refinement, the final model achieved an **AUC** of **0,85** and a **Validation Loss** of **0,42**.

All the results for each attempt and configuration used in this project can be found on the "Log" folder. The model weights were saved too and can be used to validate this results.

## Project Instructions

1. Clone the repository and navigate to the downloaded folder.
```	
git clone https://github.com/ElCasagrande/Udacity-ML-Capstone-Project.git
cd Udacity-ML-Capstone-Project
```

2. This project uses Tensorflow with GPU support. Therefore, follow [the guide](https://www.tensorflow.org/install/) to install the necessary NVIDIA software on your system.  

3. Create (and activate) a new Anaconda environment.
	- __Windows__:  
	```
	conda env create -f requirements/capstone-project-win-gpu.yml
	activate capstone-project
	```

4. Switch [Keras backend](https://keras.io/backend/) to TensorFlow.
	- __Windows__: 
	```
	set KERAS_BACKEND=tensorflow
	python -c "from keras import backend"
	```
5. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `capstone-project` environment. 
```
python -m ipykernel install --user --name capstone-project --display-name "capstone-project"
```

6. Open the notebook.
```
jupyter notebook capstone-project.ipynb
```

7. Have fun !!!