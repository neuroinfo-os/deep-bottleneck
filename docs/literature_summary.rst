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


6. SVCCA: Singular Vector Canonical Correlation Analysis
--------------------------------------------------
Maithra Raghu, Justin Gilmer, Jason Yosinski,
Jascha Sohl-Dickstein (2017). SVCCA: Singular Vector Canonical Correlation
Analysis for Deep Learning Dynamics and Interpretability.

6.1 Key Points of the paper
^^^^^^^^^^^^^^^^^^^^^^^^^^^

#. They developed a method that analyses each neuron's activation vector (i.e.
the scalar outputs that are emitted on input datapoints). This analysis gives an
insight into learning dynamics and learned representation.

#. SVCCA is a general method that compares two learned representations of
different neural network layers and architectures. It is either possible to
compare the same layer at different time steps, or simply different layers.

#. The comparison of two representations fulfills two important properties:
    * It is invariant to affine transformation (which allows the comparison
      between different layers and networks).

      * It is fast to compute, which allows more comparisons to be calculated
      than with previous methods.

6.2 How SVCCA works
^^^^^^^^^^^^^^^^^^^

#. SVCCA is short for Singular Vector Canonical Correlation Analysis and
  therefore combines the Singular Value Decomposition with a Canonical Correlation
  Analysis.

#. The representation of a neuron is defined as a table/function that maps the
  inputs on all possible outputs for a single neuron. Its representation is
  therefore studied as a set of responses over a finite set of inputs. Formally,
  that means that given a dataset :math:`X = {x_1,...,x_m}` and a neuron :math:`i`
  on layer :math:`l`, we define :math:`z^{l}_{i}` to be the vector of outputs on
  :math:`X`, i.e.

    .. math::

      z^{l}_{i} = (z^{l}_{i}(x_1),··· ,z^{l}_{i}(x_m)).

  Note that :math:`z^{l}_{i}` is a single neuron's response over the entire
  dataset and not an entire layer's response for a single input. In this sense
  the neuron can be tought of as a single vector in a high-dimensional space.
  A layer is therefore a subspace of :math:`\mathbb{R}^m` spanned by its neurons'
  vectors.

#. **Input**: takes two (note necessarily different) sets of neurons (typically
  layers of a network)

    .. math::

      l_1 = {z^{l_1}_{1}, ..., z^{l_{m_1}}_{l_1}} and l_2 = {z^{l_2}_{1}, ..., z^{l_{m_2}}_{l_2}}

#. **Step 1**: Use SVD of each  subspace to get sub-subspaces :math:`l_1' \in l_1`
  and :math:`l_2' \in l_2`, which contain of the most important directions of the
  original subspaces :math:`l_1, l_2`.

#. **Step 2**: Compute Canonical Correlation similarity of :math:`l_1', l_2'`:
  linearly transform :math:`l_1', l_2'` to be as aligned as possible and compute
  correlation coefficients.

#. **Output**: pairs of aligned directions :math:`(\~{z}_i^{l_1}, \~{z}_i^{l_2})`
  and how well their correlate :math:`\rho_i`. The SVCCA similarity is defined as

    .. math::
      \={\rho} = \frac{1}{\min(m_1,m_2)} \sum_i \rho_i .
