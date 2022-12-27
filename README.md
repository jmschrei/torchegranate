<img src="https://github.com/jmschrei/pomegranate/blob/master/docs/logo/pomegranate-logo.png" width=300>

![](https://github.com/jmschrei/torchegranate/actions/workflows/python-package.yml/badge.svg)

> **Warning**
> torchegranate is currently under rapid active development and the API has not yet been finalized. Major changes will be noted in the CHANGELOG. 

> **Note**
> This is a temporary repository that will host code until the main functionality of pomegranate is reproduced. Then, the code here will be merged back into the main pomegranate repo.

torchegranate is a rewrite of the ![pomegranate](https://github.com/jmschrei/pomegranate) library to use PyTorch as a backend. It implements probabilistic models with a modular implementation, enabling greater flexibility in terms of model creation than most models allow. Specifically, one can drop any probability distribution into any compositional model, e.g., drop Poisson distributions into a mixture model, to create any model desired without needing to explicitly hardcode each potential model. Because one is defining the distributions to use in each of the compositional models, there is no limitation on the models being homogenous -- one can create a mixture of a exponential distribution and a gamma distribution just as easily as creating a mixture entirely composed of gamma distributions. 

But that's not all! A core aspect of pomegranate's philosophy is that every probabilistic model is a probability distribution. Hidden Markov models are simply distributions over sequences and Bayesian networks are joint probability tables that are broken down according to the conditional independences defined in a graph. Functionally, this means that one can drop a mixture model into a hidden Markov model to use as an emission just as easily as one can drop an individual distribution. As part of the roadmap, and made part in possible due to the flexibility of PyTorch, complete cross-functionality should be possible, such as a Bayes classifier with hidden Markov models or a mixture of Bayesian networks.

Special shout-out to ![NumFOCUS](https://numfocus.org/) for supporting this work with a special development grant.

### Installation

`pip install torchegranate`

### Why a Rewrite?

This rewrite was motivated by three main reasons:

- <b>Speed</b>: Native PyTorch is just as fast as the hand-tuned Cython code I wrote for pomegranate, if not significantly faster.
- <b>Community Contribution</b>: A challenge that many people faced when using pomegranate was that they could not extend it because they did not know Cython, and even if they did know it, coding in Cython is a pain. I felt this pain every time I tried adding a new feature or fixing a bug. Using PyTorch as the backend significantly reduces this problem.
- <b>Interoperability</b>: Libraries like PyTorch offer a unique ability to not just utilize their computational backends but to better integrate into existing deep learning resources. This rewrite should make it easier for people to merge probabilistic models with deep learning models.

### Roadmap

The ultimate goal is for this repository to include all of the useful features from pomegranate, at which point this repository will be merged back into the main pomegranate library. However, that is quite a far way off. Here are some milestones that I see for the next few releases.

- [x] v0.1.0: Initial draft of most models with basic feature support, only on CPUs
- [ ] v0.2.0: Addition of GPU support for all existing operations and serialization via PyTorch
- [ ] v0.3.0: Addition of missing value support for all existing algorithms
- [ ] v0.4.0: Addition of sampling algorithms for each existing method
- [ ] v0.5.0: Addition of pass-through for forward and backward algorithms to enable direct inclusion of these components into PyTorch models
- [ ] v0.6.0: Addition of Bayesian networks and factor graphs in a basic form

### Frequently Asked Questions

> Why can't we just use `torch.distributions`?

`torch.distributions` is a great implementation of the statistical characteristics of many distributions, but does not implement fitting these distributions to data or using them as components of larger functions. If all you need to do is calculate log probabilities, or sample, given parameters (perhaps as output from neural network components), `torch.distributions` is a great, simple, alternative.

> What models are implemented in torchegranate?

Currently, implementations of many distributions are included, as well as general mixture models, Bayes classifiers (including naive Bayes), hidden Markov models, and Markov chains. Bayesian networks will be added soon but are not yet included.

> How much faster is this than pomegranate?

It depends on the method being used. Most individual distributions are approximately 2-3x faster. Some distributions, such as the categorical distributions, can be over 10x faster. These will be even faster if a GPU is used.
