# MCAN

### Official pytorch implementation of the paper: "Oil Spill Detection with A Multiscale Conditional Adversarial Network under Small Data Training". The codes will be kept updating and improving. 


## Oil spill detection with small data
MCAN is an effective tool to train a model with small data.

![](imgs/teaser.PNG)


## Code

### Install dependencies

```
python -m pip install -r requirements.txt
```

This code was tested with python 3.6  

###  Train
To train MCAN model on your own training images, put the training images under Input/TrainingSet, and run

```
python main_train.py 
```
###  Test
To test MCAN model on your own test images, put the test images under Input/TestSet, and run

```
python test.py --trained_model model
```

[//]: # (We acknowledge the authors for the code released at ''https://github.com/tamarott/SinGAN'', which provides useful operations for our implementation.)

