# ML 4MOST Target Selection

This repository focus on target selection using Machine Learning Techniques in the framework of 4MOST project.
The selection aims at efficiently detect Emission Line Galaxies (ELG), Bright Galaxies (BG), Luminous Red Galaxies (LRG) and Quasar (QSO) among large photometric surveys such as DES.
To do so various implementations of Multi-Layer Perceptron were designed using the kera's and scikit-learn python libraries.
The inputs used for target selection are DES magnitudes and corresponding errors as well as infrared and near infrared magnitudes and errors (i.e WISE_w1, VHS_j, VHS_k).
The output of the classifiation MLP network has the form of an array of confidence score with 5 components corresponding to confidence value of the network for the input to blong to the 5 classes considered (i.e [Others, ELG, LRG, BG, QSO])
In addition, you will find code to perform photometric redshift regression based on photometric redshift from HSC release available in the dataset.

## Getting Started

These instructions will get you a copy of the project and allow you to run the code on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

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

Make sure to extract the catalog in the "src" folder in its original .fits format.

## Run the code

The 6 main scripts of this repository are the following :

```
dataset_generator_classification.py
classification.py
utils_classification.py
dataset_generator_regression.py
regression.py
utils_regression.py
custom_metrics_classification.py
```

More explicitely :
"dataset_generator" scripts allow to generate the dataset for training and regression purpose from the fits file provided.
"utils" scripts contain all the utilities functions used in the classification/regression scripts.
"classification" script contains the neural network implementation, training callback and evaluation for classification
"regression" script contains the neural network implementation, training callback and evaluation for regression
"custom_metrics" scripts allow to define custom metrics in the form of keras callback that can be used during training of the network to monitor performances. This script was mainly justified by the fact that keras does not include anymore f1-score in its default metric, it also allow to get more in depth review of the per class performances during training as one can define class-wise metric in the later (which is useful in highly unbalanced classification problems).

### Generate Training, Validation and Testing Datasets

To generate the training, validation and testing dataset, just run the script directly using :

```
python dataset_generator.py
```

The script allows to generate various datasets from a single catalog and the user should adapt the scripts parameters depending on the problem considered.
The script is build around the "Classification_dataset_generator" class which uses arguments given in its initialization to compute and save datasets.

If the script ran successfully, a new folder named "datasets" should have appeared in the "src" folder.
The later should have the following architecture :

```
datasets/
	<catalog_filename>/
		<constraints>/
			<classification_problem>_<zsafe_threshold>_<train/val/test>_<train_split>_<test_split>_<dataset_index>_<cv_fold_number>.fits
			...
```

where <catalog_filename>, <constraints>, <classification_problem>, <zsafe_threshold>, <train/val/test>, <train_split>, <test_split>, <dataset_index>, <cv_fold_number> are the parameters given to the "Classification_dataset_generator" object. Those parameters are introduced and defined in the comments of the dataset_generator.py script.

### Train a classification network

To train a network for classification one should run the corresponding script using :

```
python classification.py
```

The script allows to train different classifiers such as Random Forest, Support Vector Machine or Multi-Layer Perceptron.
As it is the case for the generation of the dataset, the script is built around a class : "Classification".
One should adapt the arguments in the initialization of the Classification object to decide which classifier, dataset and preprocessing techniques to use.

After execution, the script store the results and saved model with the following architecture :

```
model_ANN_classification/
	<model_index>/
		checkpoints/
			ANN_checkpoints_<number of epochs at checkpoint>-epo_<metric_score>-<metric_name>.hdf5
			...
		performances/
			val/
				Contain various plot regarding network performances on validation dataset (loss, custom metrics, 					confusion matrix, PR curves, ROC curves)
			test/
				Contain various plot regarding network performances on testing dataset (loss, custom metrics, 						confusion matrix, PR curves, ROC curves)
			post_processed/
				Contain various plot regarding network performances after post processing (loss, custom metrics, 					confusion matrix, PR curves, ROC curves)
		tsboard/
			tensoboard files if tsboard enabled
		ANN_architecture.json (model architecture saved as json format)
		ANN_parameters.json (model input parameters saved as dictionnary in json format)
		classification_inputs.json (classification object parameters used to generate ANN)
		training_data.fits (data used for model training, good to keep due to the randomness of oversampling)
		ANN_weights_<number of epochs>-epo_<metric_score>-<metric_name>.hdf5 (final model weights at end of training)
	Benchmnark_ANN_inputs.csv (Table that summarizes the inputs used in classification.py for each model)
	Benchmnark_ANN_parameters.csv (Table that summarizes the inputs used in the neural network for each model)
	Benchmnark_ANN_performances.csv (Table that summarizes the performances of each model on validation data)
```

### Train a regression network

To train a network for regression task one should run the corresponding script using :

```
python regression.py
```

The script allows to train different classifiers such as Random Forest, Support Vector Machine or Multi-Layer Perceptron.
As it is the case for the classification, the script is built around a class : "Regression".
One should adapt the arguments in the initialization of the Classification object to decide which classifier, dataset and preprocessing techniques to use.

After execution, the script store the results and saved model with the following architecture :

```
model_ANN_regression/
	<model_index>/
		checkpoints/
			ANN_checkpoints_<number of epochs at checkpoint>-epo_<metric_score>-<metric_name>.hdf5
			...
		performances/
			val/
				Contain various plot regarding network performances on validation dataset (loss, custom metrics, 					confusion matrix, PR curves, ROC curves)
			test/
				Contain various plot regarding network performances on testing dataset (loss, custom metrics, 						confusion matrix, PR curves, ROC curves)
		tsboard/
			tensoboard files if tsboard enabled
		ANN_architecture.json (model architecture saved as json format)
		ANN_parameters.json (model input parameters saved as dictionnary in json format)
		regression_inputs.json (classification object parameters used to generate ANN)
		training_data.fits (data used for model training)
		ANN_weights_<number of epochs>-epo_<metric_score>-<metric_name>.hdf5 (final model weights at end of training)
	Benchmnark_ANN_inputs.csv (Table that summarizes the inputs used in classification.py for each model)
	Benchmnark_ANN_parameters.csv (Table that summarizes the inputs used in the neural network for each model)
	Benchmnark_ANN_performances.csv (Table that summarizes the performances of each model on validation data)
```

## Authors

* **Grégoire André**

## License

This project is licensed under the GNU GENERAL PUBLIC License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* [imbalance-learn](https://imbalanced-learn.readthedocs.io/en/stable/index.html)
* [keras](https://keras.io/)
* [tensorflow](https://www.tensorflow.org/)
* [numpy](http://www.numpy.org/)
* [scikit-learn](http://scikit-learn.org)
* [astropy](http://www.astropy.org/)
* [Ian Goodfellow, Yoshua Bengio, Aaron Courville, Deep Learning](https://www.deeplearningbook.org/)
