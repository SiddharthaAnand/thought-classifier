"""
Exploratory Analysis of the textual data available for sentiment analysis.

##      Analysis of Raw data | Number of positives/negatives polarity | Need for a balanced training data
##      Visualization
##      Split into training/testing data
##      k-fold cross validation
##      Generate a model
##      Pickle/serialize the model
"""
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer


def read_and_reindex(filename=None, delimiter=None):
    if filename is not None and delimiter is not None:
        data = tsv_file_reader.convert_tsv_to_data_frame(filename=filename, delimiter=delimiter)

        ############################################
        #   Some basic information about data   ####
        ############################################
        print("Size of the dataset: %d " %(len(data['text'])))

        # Shuffle the data randomly to avoid biasness while collection of the data
        data = data.reindex(np.random.permutation(data.index))
        data = data[['text', 'polarity']]
        return data


"""
Check for imbalance in the data set during the model training phase.
"""


def visualize_target_class_frequency(data=None):
    sns.catplot(x='polarity', data=data, kind="count", height=5, aspect=1.5, palette='PuBuGn_d')
    plt.show()


def visualize_word_count_and_polarity(data=None):
    sns.catplot(x='count_words', data=data, kind="count", height=5, aspect=2.5, palette='ch:.25')
    plt.show()


"""
Clean the text:
1. Remove extra spaces and convert variable spaces to a single space.
2. Take only english alphabet characters.
"""


def clean_up_data(data=None):
    if data is not None:
        tc = text_count.TextCount()
        df_eda = tc.fit_transform(data.text)
        df_eda['polarity'] = data.polarity
        return df_eda


"""
Show stats for different target class with respect to no of words.
"""


def show_distribution(df=None, col=None):
    print('Descriptive stats for {}'.format(col))
    print('-' * (len(col) + 22))
    print(df.groupby('polarity')[col].describe())
    bins = np.arange(df[col].min(), df[col].max() + 1)
    g = sns.FacetGrid(df, col='polarity', height=5, hue='polarity', palette='ch:.25')
    g = g.map(sns.distplot, col, kde=False, norm_hist=True, bins=bins)
    plt.show()


"""
Clean text using stemming/cleaning and see the output..
"""


def text_cleaner(df=None):
    from ml_code.pre_processing import clean_text
    ctext = clean_text.CleanText()
    clean_review = ctext.fit_transform(df.text)
    # Get some random sample to see how the processed text looks like!
    print(clean_review.sample(5))
    return clean_review


"""
Check and replace empty rows with some text/null/avg values
"""


def fill_empty_reviews_with_no_text(cleaned_review=None, filler_text=None):
    empty_review_rows = cleaned_review == ''
    print('{} records have no words left after cleaning text'.format(cleaned_review[empty_review_rows].count()))
    print('-' * 22)
    cleaned_review.loc[empty_review_rows] = filler_text


"""
Get frequency of words in the reviews.
"""


def analyse_count_vectorizer_feature(cleaned_review=None):
    import collections
    import pandas as pd
    cv = CountVectorizer()
    bow = cv.fit_transform(cleaned_review)
    word_freq = dict(zip(cv.get_feature_names(), np.asarray(bow.sum(axis=0)).ravel()))
    word_counter = collections.Counter(word_freq)
    word_counter_df = pd.DataFrame(word_counter.most_common(20), columns=['word', 'frequency'])

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.barplot(x="word", y="frequency", data=word_counter_df, palette="PuBuGn_d", ax=ax)
    plt.show()


if __name__ == '__main__':
    """
    This part is used for separate analysis and running of this code.
    sys.path.insert() is being used for running this module independantly
    of other modules for testing purposes.
    
    This portion will only run when this module is specifically called.
    __name__ is equal to __main__ only for the module which is run, 
    not for the ones which are imported.
    """
    import sys
    sys.path.insert(0, '/home/sid/github/thought-classifier/')

    from ml_code.file_reader import tsv_file_reader
    from ml_code.pre_processing import text_count

    filename = "data_models/raw_data/imdb_labelled.txt"
    delimiter = r'\s{3,}'
    reindexed_data = read_and_reindex(filename=filename, delimiter=delimiter)
    # 1. visualize_target_class_frequency(reindexed_data)
    # 2. word_count_frame = clean_up_data(reindexed_data)
    # 3. visualize_word_count_and_polarity(word_count_frame)
    # show_distribution(word_count_frame, 'count_words')
    cleaned_review = text_cleaner(reindexed_data)
    fill_empty_reviews_with_no_text(cleaned_review=cleaned_review, filler_text="[no_review_here]")
    analyse_count_vectorizer_feature(cleaned_review)