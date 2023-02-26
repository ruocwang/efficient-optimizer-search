## Computation Graph based Search Engine for Neural Optimizer Search


## Env
1. robust bench:
```
pip install git+https://github.com/RobustBench/robustbench.git
```


## MNIST
```
#### search
bash mc-mnistnet.sh --constraint 1 --seed 0/1

#### eval
## direct
bash eval-mnistnet.sh --optimizer G/SGD/NAG/Adam/RMSProp

## transfer (1000 steps)
# mnistnet
bash eval-mnistnet.sh --optimizer G
# mnistnet-relu
bash eval-mnistnet.sh --extra_configs "model=MNISTNetV2_Relu" --optimizer G
# mnistnet-2layers
bash eval-mnistnet.sh --extra_configs "modelcZ=MNISTNetV2_2Layer" --optimizer G
# mnistnet-large
bash eval-mnistnet.sh --extra_configs "model=MNISTNetV2_Big" --optimizer G
```

## Conv



## TODO
[ ]. Finishing Up Document
[ ]. Environment setup
[ ]. Organize searched optimizers
[ ]. Reproducibility check
[ ]. Second round of code factorization
[ ]. MultiProcessor code


## Cltation