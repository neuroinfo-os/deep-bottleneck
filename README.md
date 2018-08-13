[![Documentation Status](https://readthedocs.org/projects/deep-bottleneck/badge/?version=latest)](http://deep-bottleneck.readthedocs.io/en/latest/?badge=latest)
[![Build status](https://travis-ci.com/neuroinfo-os/deep-bottleneck.svg?branch=master)](https://travis-ci.com/neuroinfo-os/deep-bottleneck)

# Deep Bottlneck: Understanding learning in deep neural networks with the help of information theory
This repository conatains code to reproduce and expand on the results of 
[Schwartz-Ziv and Tishby](https://arxiv.org/pdf/1703.00810.pdf) and [Saxe et al.](https://openreview.net/pdf?id=ry_WPG-A-).
It is used to investigate what role compression plays in learing in deep neural networks.

## Features
* plotting of learing dynamics in the information plane
* plotting activation histograms and single neuron activations
* different datasets and mutual information estimators
* logging experiments using Sacred

## Documentation
Extensive documentation including theoretical background and API documentation can
be found at [Read the Docs](http://deep-bottleneck.readthedocs.io/en/latest/).