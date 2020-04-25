"""
Define a transformer for the Pipeline.
"""

from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


class TextCount(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None, **fit_params):
        return self

    @staticmethod
    def count_regex(pattern, text):
        import re
        return len(re.findall(pattern, text))

    def transform(self, dataframe, **transform_params):
        count_words = dataframe.apply(lambda single_review: self.count_regex(r'\w+', single_review))
        df = pd.DataFrame({
            'count_words': count_words
        })
        return df

