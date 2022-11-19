<img src="https://github.com/jmschrei/pomegranate/blob/master/docs/logo/pomegranate-logo.png" width=300>

> **Warning**
> torchegranate is currently under rapid active development and the API has not yet been finalized. Major changes will be noted in the CHANGELOG. 

torchegranate is a modular implementation of probabilistic models that uses PyTorch as the backend. It is a rewrite of the pomegranate library that maintains its spirit but changes the API to more closely match scikit-learn and PyTorch.  

Need to train a hidden Markov model using a GPU? Brrr.

Need a loss for your neural network based on a mixture model or HMM? No problem.

Need to fit a heterogeneous mixture of different distributions? Easy.

> **Note**
> This is a temporary repository that will host code until the main functionality of pomegranate is reproduced. Then, the code here will be merged back into the main pomegranate repo.

The rewrite was motivated by three main reasons:

1. Speed: Native PyTorch can easily be just as fast as the hand-tuned Cython code I wrote for pomegranate, if not significantly faster.
2. Community Contribution: A challenge many people faced when using pomegranate was that they could not extend it because they did not use Cython. Using PyTorch as the backend significantly reduces this issue.
3. Interoperability: The components of probabilistic models can now be merged with neural networks much easier to form losses or internal latent variables. 


### Frequently Asked Questions

> Why can't we just use `torch.distributions`?

`torch.distributions` is a great implementation of the statistical characteristics of many distributions, but does not implement fitting these distributions to data or using them as components of larger functions. If all you need to do is calculate log probabilities, or sample, given parameters (perhaps as output from neural network components), `torch.distributions` is a great, simple, alternative.

> What models are implemented in torchegranate?

Currently, implementations of many distributions are included, as well as general mixture models, Bayes classifiers (including naive Bayes), hidden Markov models, and Markov chains. Bayesian networks will be added soon but are not yet included.

> How much faster is this than pomegranate?

It depends on the method being used. Most individual distributions are approximately 2-3x faster. Some distributions, such as the categorical distributions, can be over 10x faster. These will be even faster if a GPU is used.
