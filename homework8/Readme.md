# File part1_LogisticRegression.py
    sclearn LogisticRegression  Model 

# File part1_SGDClassifier.py
   sclearn  SGDClassifier model

# File part2_torch.py
    pytorch LogisticRegression_torch model

Duaring this homework, i make a conclusion that:

1) We have small dataset, wich describe haotic situation. So i have found that max accuracy can be 80-82%, but we need more features.
2) LogisticRegression more simplier model than SGDClassifier. So accuracy higher and logically understandable.  Accuracy = 78-79%
3) SGDClassifier is too sensetive and hard model for this dataset and situation. The main hyperparametr here is a random_state. Other hyperparametrs are useless because model can be easily outfit. Accurancy in that cases are higher but thats only because model can predict all results by 0 or 1, so there are no prediction here. I have tunned random state and found only 1 case where SGDClassifier make predictions logically but accuracy 0.65 and its lower than LogisticRegression
4) LogisticRegression_torch, i have reached the result BCE ≈ 0.46. The accuracy result the same with sklearn. So in comparing these 3 method i think that`s max accuracy result fot this dataset with this features.
