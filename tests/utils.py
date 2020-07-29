from sklearn.base import BaseEstimator, TransformerMixin


class PrintingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, name=''):
        self.name = name
        self.count = 0

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        print(
            f'TestTransformer {self.name} is transforming! ({self.count} '
            'times so far)'
        )
        self.count += 1
        return X
