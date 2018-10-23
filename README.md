# Project Title

This repository focus on target selection using Machine Learning Techniques in the framework of 4MOST project.
The selection aims at efficiently detect Emission Line Galaxies (ELG), Bright Galaxies (BG), Luminous Red Galaxies (LRG) and Quasar (QSO) among large photometric surveys such as DES.
To do so various classifiers such as Multi-Layer Perceptron, Random Forest or Support Vector Machine were implemented using the kera's and scikit-learn python libraries.
The inputs used for target selection are DES magnitudes and corresponding errors as well as infrared and near infrared magnitudes and errors (i.e WISE_w1, VHS_j, VHS_k).
The output of the MLP network has the form of an array of confidence score with 5 components corresponding to the 5 classes considered (i.e [Others, ELG, LRG, BG, QSO])

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

[Anaconda](https://www.anaconda.com/download/)

[Git](https://git-scm.com/downloads)

```
Give examples
```

### Installation

Clone the github repository using git command :

```
Git clone https://github.com/gregoireandre/MLP_4MOST_TargetSelection.git
```

Move inside the directory :

```
cd MLP_4MOST_TargetSelection
```

Install anaconda environement :

```
conda lastro create -f lastro.yml
```

Now that the environement is installed, you can activate it using the following command (on Linux/MacOS):

```
source activate lastro
```

or on Windows using Anaconda prompt :


```
activate lastro
```

## Dataset

The dataset (i.e catalog) used to train and test the different machine learning algortihms being too big to be stored on github, the latter can be downloaded from at the following link :

https://drive.google.com/open?id=1CUUORiH8SCtoD6nB7ST_Ha_D1SQjfq1h

Make sure to extract the ccatalog in the "src" folder in its original .fits format.

## Run the code

The 3 main scripts of this repository are the following :

```
dataset_generator.py
classification.py
evaluate.py
```

As stated in their names, those scripts alloes to respectively generate the datasets from a catalog,  train a classifier on those datasets and evaluate its performance.

### Generate Training, Validation and Testing Datasets

To generate the training, validation and testing dataset, just run the script directly using :

```
python dataset_generator.py
```

The script allows to generate various datasets from a single catalog and the user should adapt the scripts parameters depending on the problem considered.
The script parameters are defined as global variables at the end of the script, their definitions as well as their values are described in the script itself.

If the script ran successfully, a new folder named "datasets" should have appeared in the "src" folder.
The later should have the following architecture :

```
datasets/
	Catalog_filename/
		Constraints/
			<Classification problem>_<train/val/test>_<train split>_<test split>_<dataset index>_<cv fold number>.fits
			...
```

### Train an algorithm

To train an algoithm one should run the classification script using :

```
python classification.py
```

The script allows to train different classifiers such as Random Forest, Support Vector Machine or Multi-Layer Perceptron.
As it is the case for the generation of the dataset, one should adapt the bottom of the script to decide which classifier, dataset and preprocessing techniques to use.

After execution, the script store the results and saved model with the following architecture :

```
model_ANN/
	<model_index>/
		checkpoints/
			<model_name>_<epoch>_<metric_score>.hdf5
			...
		figures/
			<model_name>_<metric_score>.png
			<model_name>_loss.png
		tsboard/
			<tensoboard files>
		<model_name>.json (model architecture saved as json format)
		ann_parameters.json (model input parameters saved as dictionnary in json format)
		<model_name>_<epoch>_<metric_score>.hdf5 (final model weights at end of training)
		<model_name>_report.txt (model classification performances report on validation data)
	Benchmnark_ANN_inputs.csv (Table that summarizes the inputs used in classification.py for each model)
	Benchmnark_ANN_parameters.csv (Table that summarizes the inputs used in the neural network for each model)
	Benchmnark_ANN_performances.csv (Table that summarizes the performances of each model on validation data)
model_RF/
	TO DO
model_SVM/
	TO_DO
```

## Authors

* **Grégoire André**

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* [imbalance-learn](https://imbalanced-learn.readthedocs.io/en/stable/index.html)
* [keras](https://keras.io/)
* [tensorflow](https://www.tensorflow.org/)
* [numpy](http://www.numpy.org/)
* [scikit-learn](http://scikit-learn.org)
* [astropy](http://www.astropy.org/)
* [Ian Goodfellow, Yoshua Bengio, Aaron Courville, Deep Learning](https://www.deeplearningbook.org/)
