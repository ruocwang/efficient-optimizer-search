# Default DSL
from .neural_functions import HeuristicNeuralFunction, ListToListModule, ListToAtomModule, AtomToAtomModule, ConstToConstModule, init_neural_function
from .library_functions import StartFunction, SimpleITE, ITE, FoldFunction

# Domain-specific functions
from .operator_functions import *

# features
MNIST_feat = features()

cifar10_feat = features()
imagenet_feat = features()

products_feat = features()

cora_feat = features()
citeseer_feat = features()
pubmed_feat = features()
ppi_feat = features()

## bert finetuning
cola_feat = features()
mrpc_feat = features()
stsb_feat = features()
rte_feat = features()
wnli_feat = features()

