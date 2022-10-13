# gmm.py
# Author: Jacob Schreiber <jmschreiber91@gmail.com>

import torch
from BayesClassifier import BayesClassifier


class GeneralMixtureModel(BayesClassifier):
	def __init__(self, distributions=None, priors=None, max_iters=10, threshold=0.1):
		super(GeneralMixtureModel2, self).__init__(distributions=distributions,
			priors=priors)
		self.max_iters = max_iters
		self.threshold = threshold

	def fit(self, X, sample_weights=None):
		initial_logp = self.log_probability(X).sum()
		logp = initial_logp

		max_iters = 20

		for i in range(self.max_iters):
			self.summarize(X, sample_weights=sample_weights)
			self.from_summaries()

			last_logp = logp
			logp = self.log_probability(X).sum()

			print(logp - last_logp)
			if logp - last_logp < 0.1:
				break

	def summarize(self, X, sample_weights=None):
		e = self.predict_proba(X)
		for j, d in enumerate(self.distributions):
			d.summarize(X, e[:, j:j+1])

		self._w_sum += e.sum(dim=0)
		self._n_sum += e.shape[0]
