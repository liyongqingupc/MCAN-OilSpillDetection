# MCAN
### Official pytorch implementation of the paper: "Oil Spill Detection with A Multiscale Conditional Adversarial Network under Small Data Training"
####


## Oil spill detection with small data
With MCAN, you can train a generative model from small training data, for example:

![](imgs/teaser.PNG)


## Code

### Install dependencies

```
python -m pip install -r requirements.txt
```

This code was tested with python 3.6  

###  Train
To train MCAN model with training images, put the desire training images under Input/TrainingSet, and run

```
python main_train.py 
```




