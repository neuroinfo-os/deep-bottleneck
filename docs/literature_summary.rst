Literature Summary
==================

1. ON THE INFORMATION BOTTLENECK THEORY OF DEEP LEARNING
--------------------------------------------------------

:cite:`Andrew2017`

Key Points of the paper:
^^^^^^^^^^^^^^^^^^^^^^^^
    


* none of the following claims of Tishby (:cite:`Tishby`) holds in the general case:   
 
    #. deep networks undergo two distinct phases consisting of an initial fitting phase and a subsequent compression phase
    #. the compression phase is causally related to the excellent generalization performance of deep networks
    #. the compression phase occurs due to the diffusion-like behavior of stochastic gradient descent

|

* the osberved compression is different based on the activation function: double-sided saturating nonlinearities like tanh
  yield a compression phase, but linear activation functions and single-sided saturating nonlinearities like ReLU do not.

|

* there is no evident causal connection between compression and generalization.

|

* the compression phase, when it exists, does not arise from stochasticity in training.

|

* when an input domain consists of a subset of task-relevant and task-irrelevant information, the task-irrelevant information compress
  although the overall information about the input may monotonically increase with training time. This compression happens concurrently 
  with the fitting process rather than during a subsequent compression period.

|

Most important Experiments:
^^^^^^^^^^^^
#. Tishby's experiment reconstructed: 

    * 7 fully connected hidden layers of width 12-10-7-5-4-3-2 
    * trained with stochastic gradient descent to produce a binary classification from a 12-dimensional input
    * 256 randomly selected samples per batch
    * mutual information is calculated by binning th  output activations into 30 equal intervals between -1 and 1
    * trained on Tishby dataset
    * tanh-activation function

#. Tishby's experiment reconstructed with ReLu activation:

    * 7 fully connected hidden layers of width 12-10-7-5-4-3-2 
    * trained with stochastic gradient descent to produce a binary classification from a 12-dimensional input
    * 256 randomly selected samples per batch
    * mutual information is calculated by binning th  output activations into 30 equal intervals between -1 and 1
    * ReLu-activation function

#. Tanh-activation function on MNIST: 

    * 6 fully connected hidden layers of width 784 - 1024 - 20 - 20 - 20 - 10 
    * trained with stochastic gradient descent to produce a binary classification from a 12-dimensional input
    * non-parametric kernel density mutual information estimator
    * trained on MNIST dataset
    * tanh-activation function

#. ReLu-activation function on MNIST: 

    * 6 fully connected hidden layers of width 784 - 1024 - 20 - 20 - 20 - 10 
    * trained with stochastic gradient descent to produce a binary classification from a 12-dimensional input
    * non-parametric kernel density mutual information estimator
    * trained on MNIST dataset
    * ReLu-activation function

Presentation:
^^^^^^^^^^^^^

`Google slides link <https://docs.google.com/presentation/d/1tB-TkvULUd4QvVn5ClDRDko6q8Y1EOdaZnTX3eGtxVc/edit?usp=sharing>`_



