# Info
    I comment training function for each model and save trained models in files for easier verification. Also save training metrics


# Folder logs
    contains log files for each models training process

# Folder models
    contains already trained models

# Folder metrics

    contains saved training metrics for eache model


# 1_primitive_fcnn.py
    Base and simliest nn model for MNIST dataset
    Really simple logic and good accuracy
#  2_leackyFCNN.py
    Same model as previous but with relu activation. If learning rate is huge , model can easly overfit

# 3_leackyHeFCNN.py
    Stable model which resolve big data prediciton problems but in our case we dont have huge dataset. Consequently we have good accuracy . Thinking if we increase epoch value we can get better accuracy but we need more time

# 4_leakyFCNN_he_dropout.py
    This model contains one more hyperparamatr. Posibility to drop neuron on layer. If we got a huge value p=0.9 model can train more stable but we need  a lot of time to train it and we also need a huge dataset. For our dataset p=0.3 is a good variant. Really good nn for huge data with hard logic.

# 5_leakyFCNN_he_batchnorm.py
    This model contains BatchNorm instead of Dropout. Also stable train and also faster than dropout. But we have increased tendency to overtrain

