"""
Define a transformer for the Pipeline.
"""

from sklearn.base import BaseEstimator, TransformerMixin


class TextCleaner(BaseEstimator, TransformerMixin):

    def fit(self):
        return self

    def transform(self, X, y=None, **fit_params):
        pass
