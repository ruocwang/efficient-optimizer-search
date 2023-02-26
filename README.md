# Preface

If you are interested in Google's new automatically discovered optimizer [LION](https://arxiv.org/abs/2302.06675) and want to experiement with an optimizer search algorithm that runs on commercial GPUs, please check out this repo based on our NeurIPS 2022 paper.


# ENOS - Efficient Non-parametric Optimizer Search

Code accompanying the paper:<br>
[NeurIPS'2022] [Efficient Non-Parametric Optimizer Search for Diverse Tasks](https://arxiv.org/abs/2209.13575)<br>
[**Ruochen Wang**](ruocwang.github.io), Yuanhao Xiong, Minhao Cheng, Cho-Jui Hsieh

## Abstract

Efficient and automated design of optimizers plays a crucial role in full-stack AutoML systems. However, prior methods in optimizer search are often limited by their scalability, generability, or sample efficiency. With the goal of democratizing research and application of optimizer search, we present the first efficient, scalable and generalizable framework that can directly search on the tasks of interest. We first observe that optimizer updates are fundamentally mathematical expressions applied to the gradient. Inspired by the innate tree structure of the underlying math expressions, we re-arrange the space of optimizers into a super-tree, where each path encodes an optimizer. This way, optimizer search can be naturally formulated as a path-finding problem, allowing a variety of well-established tree traversal methods to be used as the search algorithm. We adopt an adaptation of the Monte Carlo method to tree search, equipped with rejection sampling and equivalent-form detection that leverage the characteristics of optimizer update rules to further boost the sample efficiency. We provide a diverse set of tasks to benchmark our algorithm and demonstrate that, with only 128 evaluations, the proposed framework can discover optimizers that surpass both human-designed counterparts and prior optimizer search methods.


## Env
1. robust bench:
```
pip install git+https://github.com/RobustBench/robustbench.git
```


## MNIST-NET Task
```
#### search
bash mc-mnistnet.sh --constraint 1 --seed 0/1

#### eval
## direct
bash eval-mnistnet.sh --optimizer <G/SGD/NAG/Adam/RMSProp>

## transfer (1000 steps)
# mnistnet
bash eval-mnistnet.sh --optimizer <>
# mnistnet-relu
bash eval-mnistnet.sh --extra_configs "model=MNISTNetV2_Relu" --optimizer <>
# mnistnet-2layers
bash eval-mnistnet.sh --extra_configs "modelcZ=MNISTNetV2_2Layer" --optimizer <>
# mnistnet-large
bash eval-mnistnet.sh --extra_configs "model=MNISTNetV2_Big" --optimizer <>
```


## TODO
- [ ] Finishing Up Document
- [ ] Environment setup
- [ ] Organize searched optimizers
- [ ] Reproducibility check
- [ ] Second round of code factorization
- [ ] MultiProcessor code


## Citation
If you find this project helpful, please consider citing our paper:

Thanks!

```
@inproceedings{
  ruochenwang2022ENOS,
  title={Efficient Non-Parametric Optimizer Search for Diverse Tasks},
  author={Wang, Ruochen and Xiong, Yuanhao and Cheng, Minhao and Hsieh, Cho-Jui},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2022},
  url={https://arxiv.org/abs/2209.13575}
}
```
