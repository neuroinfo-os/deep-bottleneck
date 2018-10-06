- What are artificial neural networks?
- What is entropy?
- How is entropy useful for understanding artificial neural networks?


What are artificial neural networks?
====================================
Making some sorts of artificial lives, capable of acting humanly-rational, has been a long lasting dream of mankind. We started with mechanical bodies, working solely based on laws of physics, which were mostly fun creatures rather than intelligent ones. The big leap took place as we stepped in the programmable computers era; when the focus shifted to those features of human skills which were a bit more brainy. So the results became more serious and successful. Codes started beating us in some aspects of intelligence which involve memory and speed, especially when they were tested using well, and formally, structured problems. But their Achilles Heel was the tasks that need a bit of intuition and our banal common sense. So, while codes were instructed to outperform us at solving elegant logical problems at which our brains are miserably weak, they failed to carry out some simple trivial tasks that we are able to do, even without consciously thinking about them. It was like we made an intangible creature who is actually intelligent, but in a different direction perpendicular to the direction of our intelligence. Thus, we thought if we really want something that act similar to us, we need to structure it just like ourselves. And that was the very reason for all the efforts that finally led to the realization of artificial neural networks (ANNs).

Unlike the conventional codes, which are instructed what to do, in a step-by-step way and by a human supervisor, neural networks learn stuff by observing data. It is almost the same as the way our brain learns doing intuitive tasks that we have no clear idea of exactly how and exactly when we have learned doing them; for example, a trivial task like recognizing a fire hydrant in a picture of a random street. And that was the way we chose to tackle the common sense problem of AI. So, what are these neural networks?


Perceptron
----------
Let’s start with the perceptron, which is a mathematical model of a single neuron and the plainest version of an artificial neural network: a network with one single-node layer. However, from a practical point of view, a perceptron is only a humble classifier that divide input data into two categories: the ones that cause our artificial neuron fires, and the ones that does not. The procedure is like this: the perceptron takes one or multiple real numbers as input, sums over a weighted version of them, adds a constant value, bias, to the result, and then uses it as the net input to its activation function. That is the function that calculates if the perceptron is going to be activated with the inputs or not. The perceptron uses Heaviside step function as its activation function. So the output of this function is the the perceptron output.


.. image:: https://user-images.githubusercontent.com/27868570/46575181-adaca500-c9b0-11e8-8788-ce58fe1fb5bd.png
  :alt: Perceptron


In language of math, a perceptron is a simple equation:


.. image:: http://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20H%28%5Csum_%7Bi%7Dw_ix_i&plus;b%29

where **x**\ :sub:`i` \ is the **\i**\th input, **w**\ :sub:`i` \ is the weight correspond to the **\i**\th input, **b** stands for the bias, and **H** is the Heaviside step function which will be activated with positive input:

.. image:: http://latex.codecogs.com/gif.latex?%5Cinline%20%5Cdpi%7B150%7D%20H%28z%29%3D%5Cleft%5C%7B%5Cbegin%7Bmatrix%7D%200%20%5Ctext%7B%2C%20if%20%7D%20z%20%3C%200%5C%5C1%20%5Ctext%7B%2C%20if%20%7D%20z%20%5Cgeq%200%20%5Cend%7Bmatrix%7D%5Cright.

\For the sake of neatness of the formula, we add a facial input, **x**\ :sub:`0` \, which is always equal to 1 and its weight, **w**\ :sub:`0` \, represent the bias value. Then we can rewrite the perceptron equation as:


.. image:: http://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20H%28%5Csum_%7Bi%7Dw_ix_i%29

and simplify the diagram, by removing the addition node, assuming everyone knows that the activation function will work on the summation of inputs:

.. image:: https://user-images.githubusercontent.com/27868570/46575888-71804100-c9be-11e8-872f-a53d47a80f96.png


**Hands On**

::

  If we have [.5, 3, -7] as inputs, and [4, .2, 9] as our weights, and the bias sets to 2,
  the net input to the Heaviside step function is:
  
  4(.5)+.2(3)+9(-7)+2 = -58.4
  
  And since the result is negative, the perceptron output is 0.


**Snippet**

::

  Perceptron could be easily coded. It is just a bunch of basic math operations and
  an if-else statement. Here is an example code, using Python:

.. code-block:: python 
  
  import numpy as np

  def perceptron(input_vector):
      '''
      This perceptron function takes a 3-element
      array in form of a row vector as its argument,
      and returns the output of the above described
      perceptron.
      '''

      # setting the parameters
      bias = 2
      weights = np.array([4, .2, 9])

      # calculating the net input to the HSFunction
      input = np.inner(input_vector, weights) + bias

      # implementing Heaviside step function
      if input < 0:
          output = 0
      else:
          output = 1

      return output


  input_vector = np.array([.5, 3, -7])
  print('The perceptron output is ', perceptron(input_vector))



What is entropy?
================



How is entropy useful for understanding artificial neural networks?
===================================================================
