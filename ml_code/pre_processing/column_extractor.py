from sklearn.base import BaseEstimator, TransformerMixin


class ColumnExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, cols):
        self.cols = cols

    def transform(self, X, **transform_params):
        return X[self.cols]

    def fit(self, X, y=None, **fit_params):
        return self


