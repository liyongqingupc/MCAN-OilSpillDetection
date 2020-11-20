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


### User Study
The data used for the user study can be found in the 'Downloads' folder. 

'real' folder: 50 real images, randomly picked from the [places databas](http://places.csail.mit.edu/)

'fake_high_variance' folder: random samples starting from n=N for each of the real images 

'fake_mid_variance' folder: random samples starting from n=N-1 for each of the real images 

For additional details please see section 3.1 in our [paper](https://arxiv.org/pdf/1905.01164.pdf)


