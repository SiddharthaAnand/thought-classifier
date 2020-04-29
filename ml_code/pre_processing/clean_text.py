from sklearn.base import BaseEstimator, TransformerMixin
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


class CleanText(BaseEstimator, TransformerMixin):

    def remove_digits(self, review=None):
        import re
        return re.sub('\d+', '', review)

    def to_lower(self, review=None):
        return str(review).lower()

    def remove_stopwords(self, review=None):
        stopword_list = stopwords.words('english')
        whitelist = ["n't", "not", "no"]
        words_list = str(review).split()
        clean_words = [word for word in words_list if(word not in stopword_list and
                                                      word not in whitelist
                                                      and len(word))]
        return " ".join(clean_words)

    def stemming(self, review=None):
        porter = PorterStemmer()
        word_list = str(review).split()
        stemmed_words = [porter.stem(word) for word in word_list]
        return " ".join(stemmed_words)

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, **transform_params):
        clean_X = X.apply(self.remove_digits)\
                    .apply(self.remove_stopwords)\
                    .apply(self.to_lower)\
                    .apply(self.stemming)
        return clean_X

