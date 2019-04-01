# Nucleus-Mask-RCNN
A modification of Matterport's Mask RCNN for Nucleus along with mAP

•	The dataset = download nuclei_datasets.tar.gz from kaggle which needs to be copied to the root directory

•	main.py file takes an argument based on the requirement

•	python3 main.py train 
trains the model from scratch using the resnet50 weights for the FCN layers. The model is stored in the logs file.
•	python3 main.py test
tests the model on the test folder images, and stores the output along with a submit file in the results folder inside the root directory. If you do not want to train the model yourself, use the weights I have saved after 40 epochs in the logs folder.

•	python3 main.py mAP-val
plots the Precision Recall curve for the 25 validation images from the training set manually taken by nucleus.VAL_IMAGE_IDS

•	python3 main.py mAP-train
plots the PR curve for the raining data. But if you want to check the PR curve for a partial amount (Takes less time to plot and less processing capacity), input a lower number. If not, input the number of files in training data.
