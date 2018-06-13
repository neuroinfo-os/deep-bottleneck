Literature Summary
==================

1. THE INFORMATION BOTTLENECK METHOD (Tishby 1999)
--------------------------------------------------
Tishby, N., Pereira, F. C., & Bialek, W. (2000). The information bottleneck method. arXiv preprint physics/0004057.

1.1. Glossary
^^^^^^^^^^^^^

1.2. Structure
^^^^^^^^^^^^^^

1.3. Criticisim
^^^^^^^^^^^^^^^

1.4. Todo List
^^^^^^^^^^^^^^



2. DEEP LEARNING AND THE INFORMATION BOTTLENECK PRINCIPLE (Tishby 2015)
-----------------------------------------------------------------------
Tishby, N., & Zaslavsky, N. (2015, April). Deep learning and the information bottleneck principle. In Information Theory Workshop (ITW), 2015 IEEE (pp. 1-5). IEEE.

2.1. Glossary
^^^^^^^^^^^^^

2.2. Structure
^^^^^^^^^^^^^^

2.3. Criticisim
^^^^^^^^^^^^^^^

2.4. Todo List
^^^^^^^^^^^^^^



3. OPENING THE BLACK BOX OF DEEP NEURAL NETWORKS VIA INFORMATION (Tishby 2017)
------------------------------------------------------------------------------
Shwartz-Ziv, R., & Tishby, N. (2017). Opening the black box of deep neural networks via information. arXiv preprint arXiv:1703.00810.

3.1. Glossary
^^^^^^^^^^^^^

3.2. Structure
^^^^^^^^^^^^^^

3.3. Criticisim
^^^^^^^^^^^^^^^

3.4. Todo List
^^^^^^^^^^^^^^



4. ON THE INFORMATION BOTTLENECK THEORY OF DEEP LEARNING (Saxe 2018)
--------------------------------------------------------------------
Saxe, A. M., Bansal, Y., Dapello, J., Advani, M., Kolchinsky, A., Tracey, B. D., & Cox, D. D. (2018, May). On the information bottleneck theory of deep learning. In International Conference on Learning Representations.

4.1. Glossary
^^^^^^^^^^^^^

4.2. Structure
^^^^^^^^^^^^^^

4.3. Criticisim
^^^^^^^^^^^^^^^

4.4. Todo List
^^^^^^^^^^^^^^



5. ON THE INFORMATION BOTTLENECK THEORY OF DEEP LEARNING
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



