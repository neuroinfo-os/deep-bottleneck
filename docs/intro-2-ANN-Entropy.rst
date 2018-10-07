- What are artificial neural networks?
- What is entropy?
- How is entropy useful for understanding artificial neural networks?


What are artificial neural networks?
====================================
Making some sorts of artificial lives, capable of acting humanly-rational, has been a long lasting dream of mankind. We started with mechanical bodies, working solely based on laws of physics, which were mostly fun creatures rather than intelligent ones. The big leap took place as we stepped in the programmable computers era; when the focus shifted to those features of human skills which were a bit more brainy. So the results became more serious and successful. Codes started beating us in some aspects of intelligence which involve memory and speed, especially when they were tested using well, and formally, structured problems. But their Achilles Heel was the tasks that need a bit of intuition and our banal common sense. So, while codes were instructed to outperform us at solving elegant logical problems at which our brains are miserably weak, they failed to carry out some simple trivial tasks that we are able to do, even without consciously thinking about them. It was like we made an intangible creature who is actually intelligent, but in a different direction perpendicular to the direction of our intelligence. Thus, we thought if we really want something that act similar to us, we need to structure it just like ourselves. And that was the very reason for all the efforts that finally led to the realization of artificial neural networks (ANNs).

Unlike the conventional codes, which are instructed what to do, in a step-by-step way and by a human supervisor, neural networks learn stuff by observing data. It is almost the same as the way our brain learns doing intuitive tasks that we have no clear idea of exactly how and exactly when we have learned doing them; for example, a trivial task like recognizing a fire hydrant in a picture of a random street. And that was the way we chose to tackle the common sense problem of AI. So, what are these neural networks?


Perceptron
----------
Let’s start with the *perceptron*, which is a mathematical model of a single neuron and the plainest version of an artificial neural network: a network with one single-node layer. However, from a practical point of view, a perceptron is only a humble classifier that divide input data into two categories: the ones that cause our artificial neuron fires, and the ones that does not. The procedure is like this: the perceptron takes one or multiple real numbers as input, sums over a weighted version of them, adds a constant value, bias, to the result, and then uses it as the net input to its activation function. That is the function that calculates if the perceptron is going to be activated with the inputs or not. The perceptron uses Heaviside step function as its *activation function*. So the output of this function is the the perceptron output.


.. image:: https://user-images.githubusercontent.com/27868570/46575181-adaca500-c9b0-11e8-8788-ce58fe1fb5bd.png
  :alt: Perceptron
``Fig. 1. Perceptron``

|    
|   

In language of math, a perceptron is a simple equation:

.. image:: http://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20H%28%5Csum_%7Bi%7Dw_ix_i&plus;b%29
``Eq. 1``

|    
|  

where **x**\ :sub:`i` \ is the **\i**\th input, **w**\ :sub:`i` \ is the weight correspond to the **\i**\th input, **b** stands for the bias, and **H** is the Heaviside step function which will be activated with positive input:

.. image:: http://latex.codecogs.com/gif.latex?%5Cinline%20%5Cdpi%7B150%7D%20H%28z%29%3D%5Cleft%5C%7B%5Cbegin%7Bmatrix%7D%200%20%5Ctext%7B%2C%20if%20%7D%20z%20%3C%200%5C%5C1%20%5Ctext%7B%2C%20if%20%7D%20z%20%5Cgeq%200%20%5Cend%7Bmatrix%7D%5Cright.
``Eq. 2`` [#]_

|    
|  

\For the sake of neatness of the formula, we add a facial input, **x**\ :sub:`0` \, which is always equal to 1 and its weight, **w**\ :sub:`0` \, represent the bias value. Then we can rewrite the perceptron equation as:


.. image:: http://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20H%28%5Csum_%7Bi%7Dw_ix_i%29
``Eq. 3``

|    
|  

and simplify the diagram, by removing the addition node, assuming everyone knows that the activation function will work on the summation of inputs:

.. image:: https://user-images.githubusercontent.com/27868570/46575888-71804100-c9be-11e8-872f-a53d47a80f96.png
``Fig. 2. Perceptron``

|    
|  


**Hands On (1)**

::

  If we have [.5, 3, -7] as inputs, and [4, .2, 9] as our weights, and the bias sets to 2, the net
  input to the Heaviside step function is:
  
  4(.5)+.2(3)+9(-7)+2 = -58.4
  
  And since the result is negative, the perceptron output is 0.


**Snippet (1)**

::

  Perceptron could be easily coded. It is just a bunch of basic math operations and an if-else
  statement. Here is an example code, using Python:

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


As we did with the code, dealing with a perceptron, the input is the only variable we have. But the weights and the bias are the parameters of our perceptron and parts of its architecture. It does not necessarily mean that the weights and the bias take constant values. On the contrary, we will see that the most important, and the beauty, of perceptron is its ability to learn and this learning happens through the change of the weights and the bias.

But for now, let’s just talk about what does each of the perceptron parameters do? We can use a simple example. Assume you want to use a perceptron deciding if a specific person likes watching a specific movie or not.\ [#]_ You could define an almost arbitrary set of criteria as your perceptron input, like the movie genre, how good are the *ctors, and say the movie production budget. We can quantize these three criteria assuming the person loves watching comedies, so if the movie genre is comedy (1) or not (0). And the total number of prestigious awards won by the four leading/supporting actors, and the budget in million USD. The output 0 means the person, probably, does not like the movie and 1 means she, probably, does.

.. image:: https://user-images.githubusercontent.com/27868570/46581161-bc886b80-ca33-11e8-88fa-cbf9ffafe517.png
``Fig. 3. A perceptron for binary classification of movies for a single Netflix user``

|    
|  

Now it is easier to have an intuitive understanding of what each of perceptron parameters does. Weights help to give a more important factor, a heavier effect on the final decision. So for example, if the person is a huge fan of glorious fantasy movies with heavy CGI, we have to set **w**\ :sub:`1` \ a little bit higher. Or if she is open to discovering new talents over watching the same arrogant acting styles, we could lower down **w**\ :sub:`2` \ a bit. 
The bias role, however, is not as obvious as the weights. The simplest explanation is that bias shift the firing threshold of the perceptron or to be accurate the activation function. Suppose the intended person cares equally for the three elements of input and won’t watch a movie that fails to meet each one them. Then we have to set the bias so high that a high score in none of these three indices cannot make the perceptron fire, singly. Or if she probably would like Hobbit-kinds of movie, even though they do not fit in comedy genre, we can lower down the bias to the extent that having high scores, the *Actors* and the *Budget* could fire the perceptron together. You might think that we could do all these kind of arrangements solely using the weights. So let’s deal with this case in which all the input parameters are equal to zero. Without adding a bias term the output would be zero regardless of what we are taking in, and what we are willing to classify.


**Hands On (2)**

::

  Assume we have two binary inputs, A and B, which could be either 0 or 1. What we want is to
  design a perceptron that takes A and B and behaves like a NOR gate; that is the perceptron
  output will be 1 if and only if both A and B are 0, otherwise the output will be 0.

  It is not always guaranteed for all problems, but in this case, we could do the design in too
  many different ways, with a wide variety of values as weights and the bias. One possible valid
  combination of the parameters is: wA = -2, wB = -1, and the bias = 1. We can check the results:
  
.. image:: https://user-images.githubusercontent.com/27868570/46581680-1e010800-ca3d-11e8-8c83-945878afe6bd.png

::

  Another valid set of parameters would be: wA = -0.5, wB = -0.5, and .4 for the bias. You can
  think of many more sets of valid parameters yourself.
  
  Now try designing this perceptron without adding bias.


The last thing to talk about is the activation function. The function is like the perceptron brain. Even though it does not do complicated calculations, but without it the perceptron is nothing but a linear combination of the inputs.\ [#]_ The activation function helps perceptron to learn. Once the perceptron parameters are set, it is able to differentiate between different sets of inputs  and to make decisions via its elementary mechanism of ‘fire’ or ‘do not fire’.

That would be also fun to compare a perceptron with a neuron; provided that you do not take this comparison too seriously.\ [#]_ You can think of the inputs, naïvely, as chemoelectrical signals transmitting through dendrites (weights), reaching the neuron (Heaviside step function), if the pulse passes the threshold (bias), the neuron fires down the axon (the output is 1), otherwise it does not (the output is 0). 


The Network
-----------
So… not a big deal? We have a basic classifier which it is limited to linearly separable data. Suppose we want to divide a set of samples that are, somehow, represented using a coordinate system. The perceptron would be able to do the task, if and only if, the two sets could be separated by drawing a single straight line between them.\ [#]_

**Problem (1)**
::

  Design a perceptron that takes two binary inputs, A and B and returns the XOR value of them:
  
.. image::  https://user-images.githubusercontent.com/27868570/46582158-2b20f580-ca43-11e8-8d15-4ae0779c5a37.png
|    

So at this point, perceptron might seem a little boring. But we can make it wildly exciting with taking one step further in imitating our brain structure by connecting artificial neurons together to form a network in which each perceptron output is fed as input to another perceptron; something like this:

.. image:: https://user-images.githubusercontent.com/27868570/46582293-97e8bf80-ca44-11e8-9dae-832699152ee2.png
``Fig. 4. An artificial neural network``

|    
|  

As you see in the picture, the artificial neurons, or simply the nodes, are organized in layers. Nodes in a layer are not connected to each other. They are just connected to other nodes in their previous and/or next layers, except for the bias nodes. The bias nodes are not connected to their previous layer nodes, because being connected backward means their value is going to be set with the incoming flow. But bias nodes, as we see in perceptron, are conventionally set to feed 1,\ [#]_ so they are disconnected from their previous layers.

The first layer of the network is the input layer, and the last one is the output layer. Every layer in between is called a hidden layer. Note that, in the above picture, the input layer is more of a decorative setting, or a placeholder only to represent the input flow. The nodes in this layer are not actual perceptrons. They, just like the bias nodes, merely stand for input variables, and unlike the other nodes in the network, do not represent any activation function.\ [#]_ When we are counting a network layers, we only consider the layers with adjustable weights led to them. So in this case, we do not count the input layer and say it is a 2-layer neural network, or the depth of this network is 2. The number of neurons in each layer is called its width. But, just like the poor input layer, we do not include bias nodes while counting the width. So in our network the hidden layer width is 4 and the output layer width is 2.

As the depth of the network increases, it could easier deal with the more complicated patterns. The same happens when the width of layers grows. What this complex structure does is to break down the input data into small fragments and find a way to combine the most informative parts as output.

Imagine we want to estimate people income, based on their age, education, and say blood pressure. Assume we want to use the multiple linear regression method to accomplish the task. So what we do is to find how much and in which way each of our explanatory variables (i.e. age, education, and blood pressure) affects the income. That is, we reduce income to summation of our variables multiplied by their corresponding coefficient plus a bias term. Sounds good, does not work all the time. What we neglect here is the implicit relations between the explanatory variables, themselves. Like the general fact that, as people age, their blood pressure increases. Now what a neural network with its hidden layers does is to taking these relations into account. How? With chopping each input variable into pieces, thanks to many nodes in a one single layer, and letting these pieces each of which belongs to a different variable, combine together with a specific proportion, set by the weights, in the next layer. In other word, a neural network let the input variable have interaction with each other. And that is how the increase of width and the depth enable the network to handle and to construct more complex data structures.

**Problem (2)**

::

  We discussed a privilege of neural networks over the multiple linear regression in doing a specific
  task. Regarding the same task, would the neural network performance still have any privilege over a
  multivariate nonlinear regression, which can handle nonlinear dependency of a variable on multiple
  explanatory variables?

**Snippet (2)**

::
  
  Assume we have the following network, in which all the nodes in the hidden and output layers have
  Heaviside step function as their activation function:

.. image:: https://user-images.githubusercontent.com/27868570/46582663-cbc6e380-ca4a-11e8-806e-8332f6daa22a.png

::

  The hidden layer weights are given with the following connectivity matrix: 

.. image:: http://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%5C%20%5C%20%5C%20%5C%20%5Cbegin%7Bmatrix%7D%20x_1%26%20x_2%26%20x_3%20%5Cend%7Bmatrix%7D%20%5C%5C%20%5Cbegin%7Bmatrix%7D%20h_1%5C%5C%20h_2%5C%5C%20h_3%5C%5C%20h_4%20%5Cend%7Bmatrix%7D%20%5Cbegin%7Bbmatrix%7D%204%263%262%20%5C%5C%20-2%261%26.5%20%5C%5C%202%26-5%261.2%5C%5C%203%26-1%266%20%5Cend%7Bbmatrix%7D

::

  So according to this matrix, w32 or the weight between the second input x2 and the third node in the
  hidden layer, h3, is 5. That is, x2 will be multiplied by -5, before being fed to h3. You might feel
  a little uncomfortable with w32 convention of labeling and like w23 much better. But you will see
  noting the destination layer index before the origin layer makes life much easier. In addition, you
  can always remember that the weights are set only to adjust the value which is going to be fed to
  the next layer.
  
  And, in the same way, the following connectivity matrix gives us the output layer weights: 
  
.. image:: http://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%5C%20%5C%20%5C%20%5C%20%5Cbegin%7Bmatrix%7D%20h_1%26%20h_2%26%20h_3%26%20h_4%20%5Cend%7Bmatrix%7D%20%5C%5C%20%5Cbegin%7Bmatrix%7D%20y_1%5C%5C%20y_2%20%5Cend%7Bmatrix%7D%20%5Cbegin%7Bbmatrix%7D%202%26-1%265%263.2%20%5C%5C%20-4.5%261%263%262%20%5Cend%7Bbmatrix%7D

::

  And the bias vectors are:  
  
.. image:: http://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20B_0%20%3D%20%5Cbegin%7Bbmatrix%7D%202%20%5C%5C%20-3%5C%5C%201%5C%5C%20.6%5C%5C%20%5Cend%7Bbmatrix%7D%20B_1%20%3D%20%5Cbegin%7Bbmatrix%7D%204%20%5C%5C%205%20%5Cend%7Bbmatrix%7D


::

  Now we want to write a code to model this network, get a numpy array with the shape of (3,) as the
  input and returns the network output:  


.. code-block:: python 

  import numpy as np

  # Modeling Heaviside Step function
  def heaviside(z):
      '''
      This function models the Heaviside Step Function;
      it takes z, a real number, and returns 0 if it is
      a negative number, else returns 1.
      '''
      if z < 0:
          return 0
      else:
          return 1

  # And vectorizing it, suitable for applying element-wise
  heaviside_vec = np.vectorize(heaviside)

  def ann(input_0):
      '''
      This Artificial Neural Network function takes a 3-element
      array in the form of a row vector as its argument, and returns
      a two-element row vector as its output.
      '''

      # setting the parameters
      bias_0 = np.array([2, -3, 1, .6])
      bias_1 = np.array([4, 5])
      weights_10 = np.array([[4, 3, 2], [-2, 1, .5], [2, -5, 1.2], [3, -1, 6]])
      weights_21 = np.array([[2, -1, 5, 3.2], [-4.5, 1, 3, 2]])

      # calculating the net input to the first (hidden) layer
      input_1 = np.matmul(weights_10, input_0.transpose()) + bias_0.transpose()

      # calculating the output of the first (hidden) layer
      output_1 = heaviside_vec(input_1)

      # calculating the net input to the second (output) layer
      input_2 = np.matmul(weights_21, output_1.transpose()) + bias_1.transpose()

      # calculating the output of to the second (output) layer
      output_2 = heaviside_vec(input_2)

      return output_2

  
.. image::
.. image::
.. image::
.. image::
.. image::
.. image::
.. image::
.. image::
.. image::

What is entropy?
================

|    
|  


.. image:: https://images.mysafetysign.com/img/lg/K/Slow-Construction-Area-Sign-K-5798.gif

|    
|  



How is entropy useful for understanding artificial neural networks?
===================================================================

|    
|  



.. image:: https://images-na.ssl-images-amazon.com/images/I/410zfLWCuTL.jpg

|    
|  



.. [#] We usually denote an activation function input with the letter z, rather than good old x, in order to prevent any confusion of the function input with the perceptron/network inputs.
.. [#] For motivation, assume Netflix offered a US$1,000,000 prize for designing this perceptron.
.. [#] Plus bias which in no-activation-function case, is itself an irrelevant factor.
.. [#] Yes, the original idea was to imitate the way our brain works, but let’s be honest with ourselves, do we know how our brain works? But that aside, perceptron and ANNs have adopted a couple of important and effective macro features of our brain structure, like not being a simple/linear transmitter but getting activated with specific functions/patterns or the network structure itself which is made up of, generally, uniform elements.
.. [#] Or a plane/hyperplane for 3 and more dimensions.
.. [#] The value 1 is arbitrary, and only more convenient to work with. But whatever other value you assign to the bias nodes it should be constant during the flow of data through the network. 
.. [#] However, we will see that this is not a rule.
