# StarGAN-v2
Advanced Machine LEarning Project : Reimplementing StarGANv2


This repository is split into 4 main folders : 
	
**architecture** :
Contains the scripts used to implement the StarGANv2 model and some testing files

**dataloader** :
Contains the script Dataloader.py containing the different classes and fucntions needed for the data management during training and evaluation

**train** :
Folder containing the Trainer.py script, in which there is the Trainer class handling the training procedure

**evaluation** :
This folder contains two scripts : EvluationMetrics.py and Evaluator.py. EvaluationMetrics handles the evaluation with the FID metric (needed hard coded implementation) and Evaluator handles the evaluation protocol for LPIPS (external module used : lpips) and FID.

-------

The results of the evaluation are in the metrics folder. There is  **LPIPS.txt** file for the LPIPS metrics and 2 folders for each model's FID in a json format (celeba_hq trained model and afhq trained model)

In order to recreate the results there are 2 Jupyter Notebooks : train_test and eval_test. The first is to train the model and the latter for evaluation. Those scripts have a params cell dedicated handling all the hyperparameters of training, the paths to datasets and other parameters. The pretrained weights of the two models can be downloaded form this link https://drive.google.com/drive/u/6/folders/12wFJIVgQR2emESxrYf3ML9exeje15126, so you don't need to train the model again, just load the last checkpoint to the model following the eval_test.ipynb protocol and you'll be good to go !

The last Jupyter Notebook **results.ipynb** is there to generate the figures and images used in the project report.


Contributors : Yang Li, Younes Hamai, Khaled Guedouah and Balthazar Bujard

  ...
