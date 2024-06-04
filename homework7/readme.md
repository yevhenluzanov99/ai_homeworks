

# File functions.py
    graph function to plot graphic by formula
    
# File part1_LinearRegression.py
    sclearn LinearRegression  Model 

# File part2_SGDRegressor.py
   sclearn  SGDRegressor model

# File part3_torch.py
    torch Linear model and child class LinearRegression_torch 





Duaring this homework, i make a conclusion that:
1) Firstly we comparing two methods of lib scklearn linear_model.LinearRegression and linear_model.SGDRegressor  
They use different method for searching local or global minimums. LinearRegression uses algoritm of Least Squares Optimization
and SGDRegressor uses method of Stochastic Gradient Descent. In our case we dont see any differences because we have small dataset with only one feature and intuitively linear target. I think if we have huge dataset we shouldn`t use  LinearRegression because it costs too expensive. 
2) After this i explored lib pytorch and his class Linear. I tried to tune hyperparametrs and find best of them. So i have next variants of hyperparametrs:

    2.1) 1500-2000 epochs, batch_size=200, optimizer= torch.optim.SGD(model.parameters(), lr=0.00001)
    Train Epoch: 1544        [Train Loss]: 657.785535        [Test Loss]: 600.987915
    this optimiser is too slow. if we have harder dataset this optimiser is useless. but for easy dataset it`s smoother. Also learning rate is too sensetive, i have found only lr=0.00001 wich dont waste all my time.
    2.2) 500-550 epochs, batch_size=200,  optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, lr=0.00001)
    Train Epoch: 499         [Train Loss]: 686.764217        [Test Loss]: 594.048279
    this optimiser is upgrade version of SGD but here we use momentum wich improve our learning rate. So we have better and faster result, but the initial learning rate also too sensetive
    2.3)100 epochs, batch_size=200-500, optimizer = Adagrad  lr=1
    Train Epoch: 99          [Train Loss]: 646.585424        [Test Loss]: 595.312805
    this optimiser use Gradient Descent and accumulate him. Better and Faster then privious optimiser. Learning rate is not sensetive but it must be in scopes. I think we need to have a bit higher learning rate because  it fades away quickly.
    2.4)50-100 epochs, batch_size=200,  optimizer = RMSprop,  lr=0.01
    Train Epoch: 42          [Train Loss]: 657.670685        [Test Loss]: 600.677429
    Train Epoch: 99          [Train Loss]: 661.395138        [Test Loss]: 731.329834
    this optimiser fix problem of Adagrad and scale gradient from Adagrad. So as a result we fix problem if quick fades away of learning rate.
    2.5) optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    Train Epoch: 99          [Train Loss]: 666.461413        [Test Loss]: 598.716858
    best optimiser , containts momentum for learning rate and gradient scaling from  RMSprop