
Glossary
========

Information Theory Basics
-------------------------
This glossary is mainly based on MacKay's *Information Theory, Inference
and Learning Algorithms*.
If not marked otherwise, all information below can be found there.

Prerequisites: random variable, probability distribution

One major question in information theory is "how to measure information content",
"how to compress data" and "how to communicate perfectly over imperfect communcation
channels".

At first we will introduce some basic definitions.

The Shannon Information Content
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The Shannon information content of an outcome :math:`{x}` is defined as

.. math::

  h(x) = \log_2 \frac{1}{P(x)}.

The unit of this measurement is called "bits", which does not allude to 0s and 1s.


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
Notice that entropy is defined as the average Shannon information content of an
outcome and therefore is also measured in bits.


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
We introduce the "Asymptotic equipartion" principle which can be seen as a *law
of large numbers*. This principle denotes that for an ensemble of :math:`N` independent
and identically distributed (i.i.d.) random variables :math:`X^N \equiv (X_1, X_2, \dots, X_N)`,
with :math:`N` sufficiently large, the outcome :math:`x = (x_1, x_2, \dots , x_N)`
is almost certain to belong to a subset of :math:`\mathcal{A}_X^N` with :math:`2^{NH(X)}`
members, each having a probability that is 'close to'  :math:`2^{-NH(X)}`.

The typical set is defined as

.. math::

  T_{N \beta} \equiv \{ x \in \mathcal{A}_X^N : | \frac{1}{N} \log_2 \frac{1}{P(x)} - H | < \beta \}.

The parameter :math:`\beta` sets how close the probability has to be to :math:`2^{-NH}`
in order to call an element part of the typical set, :math:`\mathcal{A}_X` is the
alphabet for an arbitrary ensemble :math:`X`.

Shannon's Source Coding Theorem
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^




Mathematical Terms in Tishby's Experiments
------------------------------------------

Stochastic Gradient Descent
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Spherical Harmonic power spectrum [Tishby (2017) 3.1 Experimental setup]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TODO

O(3) rotations of the sphere [Tishby (2017) 3.1 Experimental setup]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TODO
