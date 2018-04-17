
Glossary
========

Information Theory Basics
-------------------------

Prerequisites: random variable, probability distribution

..
  Ensemble
  ^^^^^^^^
  We extend the notion of a random variable to the notion of an **ensemble**.
  An **ensemble** :math:`{X}` is a triplet :math:`(x, A_X, P_X)`, where :math:`x`
  is just the variable denoting the outcomes of the random variable, :math:`A_X`
  is the set of all possible outcomes and :math:`P_X` is the defining probability
  distribution.



Entropy
^^^^^^^
Let :math:`X` be an random variable and :math:`A_X` the set of possible
outcomes. The entropy is defined as

.. math::

  H(X) = \sum_{x} p(x) \ log_2\left(\frac{1}{p(x)}\right).

The **entropy** describes how much we know about the outcome before
the experiment. This means

.. math::

  H(X) = 0 &\iff \text{One outcome has a probability of 100%.} \\
  H(X) = log_2(|A_X|) &\iff \text{All events have the same probability.}


The **entropy** is bounded from above and below: :math:`0 \leq H(X) \leq log_2(|A_X|)`.

Entropy for two dependent variables
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Let :math:`X` and :math:`Y` be two dependent random variables.

**joint entropy**

.. math::
  H(X,Y) = \sum_{x,y} p(x,y) \ log_2\left(\frac{1}{p(x,y)}\right)

**conditional entropy if one variable is observed**

.. math::
  H(X|y=b) = \sum_{x} p(x|y=b) \ log_2\left(\frac{1}{p(x|y=b)}\right)

**conditional entropy in general**

.. math::
  H(X|Y) &= \sum_{y} p(y) \ H(X|y=y) \\
         &= \sum_{x,y} p(x,y) \ log_2\left(\frac{1}{p(x|y)}\right)

**chain rule for entropy**

.. math::
  H(X,Y) &= H(X) + H(Y|X) \\
         &= H(Y) + H(X|Y)


Mutual Information
^^^^^^^^^^^^^^^^^^
Let :math:`X` and :math:`Y` be two random variables. The **mutual information**
between these variables is then defined as

.. math::
  I(X;Y) &= H(X) - H(X|Y) \\
         &= H(Y) - H(Y|X) \\
         &= H(X) + H(Y) - H(X,Y)

The **mutual information** describes how much uncertainty about the one variable
remains if we observe the other. It holds that

.. math::
  I(X;Y) = I(Y;X)
  I(X;Y) \geq 0

The following figure gives a good overview:

.. figure:: _static/images/mutual_info_overview.png

    Mutual information overview.


Kullback-Leibler divergence
^^^^^^^^^^^^^^^^^^^^^^^^^^^
Let :math:`X` be a random variable and :math:`p(x)` and :math:`q(x)` two
probability distributions over this random variable. The **Kullback-Leibler
divergence** is defined as

.. math::

  D_{KL}(p||q) = \sum_{x} p(x) \ log_2\left(\frac{p(x)}{q(x)}\right)


The **Kullback-Leibler divergence** is often called *relative entropy* and
denotes "something" like a distance between two distributions:

.. math::

  &D_{KL}(p||q) \geq 0 \\
  &D_{KL}(p||q) = 0 \iff p=q

Yet it is not a real distance as symmetry is not given.



Typicality
^^^^^^^^^^
TODO



Mathematical Terms in Tishby's Experiments
------------------------------------------

Spherical Harmonic power spectrum [Tishby (2017) 3.1 Experimental setup]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TODO

O(3) rotations of the sphere [Tishby (2017) 3.1 Experimental setup]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TODO
