"""
Exploratory Analysis of the textual data available for sentiment analysis.

##      Analysis of Raw data | Number of positives/negatives polarity | Need for a balanced training data
##      Visualization of word count frequency | Visualization of word count after cleaning
##      Split into training/testing data
##      k-fold cross validation
##      Use different kinds of feature selection (countvectorizer, word to vector, etc)
##      Use different models for model comparison
##      Generate a model
##      Pickle/serialize the model
"""
import sys
sys.path.insert(0, '/home/sid/github/thought-classifier/')
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from ml_code.pre_processing import column_extractor
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from pprint import pprint
from time import time
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from ml_code.pre_processing import clean_text
from ml_code.pre_processing import text_count
from ml_code.file_reader import tsv_file_reader



df_model = None
def read_and_reindex(filename=None, delimiter=None):
    """
    Read and reindex data to avoid data writing/storage biases.

    :param filename:
    :param delimiter:
    :return:
    """
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


def visualize_target_class_frequency(data=None):
    """
    Check for imbalance in the data set during the model training phase.

    :param data:
    :return:
    """
    sns.catplot(x='polarity', data=data, kind="count", height=5, aspect=1.5, palette='PuBuGn_d')
    plt.show()


def visualize_word_count_and_polarity(data=None):
    sns.catplot(x='count_words', data=data, kind="count", height=5, aspect=2.5, palette='ch:.25')
    plt.show()


def clean_up_data(data=None):
    """
    Clean the text:
    1. Remove extra spaces and convert variable spaces to a single space.
    2. Take only english alphabet characters.
    :param data:
    :return:
    """
    if data is not None:
        tc = text_count.TextCount()
        df_eda = tc.fit_transform(data.text)
        df_eda['polarity'] = data.polarity
        return df_eda


def show_distribution(df=None, col=None):
    """
    Show stats for different target class with respect to no of words.
    :param df:
    :param col:
    :return:
    """
    print('Descriptive stats for {}'.format(col))
    print('-' * (len(col) + 22))
    print(df.groupby('polarity')[col].describe())
    bins = np.arange(df[col].min(), df[col].max() + 1)
    g = sns.FacetGrid(df, col='polarity', height=5, hue='polarity', palette='ch:.25')
    g = g.map(sns.distplot, col, kde=False, norm_hist=True, bins=bins)
    plt.show()


def text_cleaner(df=None):
    """
    Clean text using stemming/cleaning and see the output..
    :param df:
    :return:
    """
    ctext = clean_text.CleanText()
    clean_review = ctext.fit_transform(df.text)
    # Get some random sample to see how the processed text looks like!
    print(clean_review.sample(5))
    return clean_review


def fill_empty_reviews_with_no_text(cleaned_review=None, filler_text=None):
    """
    Check and replace empty rows with some text/null/avg values
    :param cleaned_review:
    :param filler_text:
    :return:
    """
    empty_review_rows = cleaned_review == ''
    print('{} records have no words left after cleaning text'.format(cleaned_review[empty_review_rows].count()))
    print('-' * 22)
    cleaned_review.loc[empty_review_rows] = filler_text
    return cleaned_review


def analyse_count_vectorizer_feature(cleaned_review=None):
    """
    Get frequency of words in the reviews.
    :param cleaned_review:
    :return:
    """
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


def create_test_data(df_eda=None, sr_clean=None):
    """
    Create training/testing data before the training begins.
    :param df_eda: One dataframe which consists of the count values for every review
    :param sr_clean: Another data frame which consists of the cleaned values for every review
    :return: training and testing split data
    """
    global df_model
    df_model = df_eda
    df_model['clean_text'] = sr_clean
    df_model.columns.tolist()
    print(df_eda)
    return train_test_split(df_model.drop('polarity', axis=1), df_model.polarity, test_size=0.1, random_state=37)


def grid_vect(clf, parameters_clf, X_train, X_test, parameters_text=None, vect=None, is_w2v=None):
    """
    Apply grid search for finding the best hyper parameters and accuracy of that model.
    :param clf: classifier
    :param parameters_clf: parameters of the classifier
    :param X_train: train data
    :param X_test: test data
    :param parameters_text:
    :param vect:
    :param is_w2v:
    :return: Results: Accuracy of the model
    """
    text_count_col = ['count_words']
    SIZE = 50
    if is_w2v:
        w2v_cols = []
        for i in range(SIZE):
            w2v_cols.append(i)
        features = FeatureUnion([('textcount', column_extractor.ColumnExtractor(cols=text_count_col)),
                                 ('w2v', column_extractor.ColumnExtractor(cols=w2v_cols))],
                                n_jobs=-1)
    else:
        features = FeatureUnion([('textcount', column_extractor.ColumnExtractor(cols=text_count_col)),
                                 ('pipe', Pipeline([('cleantext', column_extractor.ColumnExtractor(cols='clean_text')),
                                                    ('vect', vect)]))],
                                n_jobs=-1)
    pipeline = Pipeline([
        ('features', features),
        ('clf', clf)
    ])

    parameters = dict()
    if parameters_text:
        parameters.update(parameters_text)
    parameters.update(parameters_clf)

    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1, cv=5)
    print("Performing grid search...")
    print("pipeline:", [name for name, _ in pipeline.steps])
    print("parameters:")
    pprint(parameters)

    t0 = time()
    grid_search.fit(X_train, y_train)
    print("done in %0.3fs" %(time() - t0))
    print()

    print("Best CV score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

    print("Test score with best_estimator_: %0.3f" % grid_search.best_estimator_.score(X_test, y_test))
    print("\n")
    print("Classification Report Test Data")
    print(classification_report(y_test, grid_search.best_estimator_.predict(X_test)))

    return grid_search


def seed_model_before_start():
    # Parameter grid settings for the vectorizers (Count and TFIDF)
    parameters_vect = {
        'features__pipe__vect__max_df': (0.25, 0.5, 0.75),
        'features__pipe__vect__ngram_range': ((1, 1), (1, 2)),
        'features__pipe__vect__min_df': (1, 2)
    }
    # Parameter grid settings for MultinomialNB
    parameters_mnb = {
        'clf__alpha': (0.25, 0.5, 0.75)
    }
    # Parameter grid settings for LogisticRegression
    parameters_logreg = {
        'clf__C': (0.25, 0.5, 1.0),
        'clf__penalty': ('l1', 'l2')
    }
    return parameters_mnb, parameters_vect, parameters_logreg


def find_model_using_gridsearch(parameters_mnb=None, parameters_vect=None, parameters_logreg=None, data=None):
    """
    Use the hyper parameters to find a classifier and a feature set which gives the best result.
    :param parameters_mnb:
    :param parameters_vect:
    :param parameters_logreg:
    :return:
    """
    mnb = MultinomialNB()
    logreg = LogisticRegression(C=1, penalty='l1', solver='liblinear', dual=False)

    ####################################################################################
    #              Taking word count vectorizer as one of the feature vectors          #
    ####################################################################################
    countvect = CountVectorizer()
    # MultinomialNB
    # best_mnb_countvect = grid_vect(mnb,
    #                                parameters_mnb,
    #                                X_train,
    #                                X_test,
    #                                parameters_text=parameters_vect,
    #                                vect=countvect)
    # joblib.dump(best_mnb_countvect, 'ml_code/output/best_mnb_countvect.pkl')
    # LogisticRegression
    best_logreg_countvect = grid_vect(logreg,
                                      parameters_logreg,
                                      data['X_train'],
                                      data['X_test'],
                                      parameters_text=parameters_vect,
                                      vect=countvect)
    joblib.dump(best_logreg_countvect, 'ml_code/output/best_logreg_countvect.pkl')
    return best_logreg_countvect

    ####################################################################################
    #                 Taking TF-IDF as one of the feature vectors                      #
    ####################################################################################
    # TF-IDF Vector
    # tfidfvect = TfidfVectorizer()
    # MultinomialNB
    # best_mnb_tfidf = grid_vect(mnb,
    #                            parameters_mnb,
    #                            X_train,
    #                            X_test,
    #                            parameters_text=parameters_vect,
    #                            vect=tfidfvect)
    # joblib.dump(best_mnb_tfidf, 'ml_code/output/best_mnb_tfidf.pkl')
    # LogisticRegression
    # best_logreg_tfidf = grid_vect(logreg,
    #                               parameters_mnb,
    #                               X_train,
    #                               X_test,
    #                               parameters_text=parameters_vect,
    #                               vect=tfidfvect)
    # joblib.dump(best_logreg_tfidf, 'ml_code/output/best_logreg_tfidf.pkl')


def predict_sentiment(text=None):
    """
    This method creates a model using logistic regression with some defined parameters using a gridsearch method
    to find the best possible parameters.
    It predicts the sentiment at runtime after that.
    It serializes the model into a file using joblib.
    :param best_mnb_countvect:
    :param text:
    :return: None
    """
    global df_model
    # word_count_frame = clean_up_data(text)
    textcountscols = ['count_words']
    # w2v_cols = 2
    features = FeatureUnion([('textcounts', column_extractor.ColumnExtractor(cols=textcountscols))
                                , ('pipe', Pipeline([('cleantext', column_extractor.ColumnExtractor(cols='clean_text'))
                                                        , ('vect', CountVectorizer(max_df=0.5, min_df=1,
                                                                                   ngram_range=(1, 2)))]))]
                            , n_jobs=-1)
    pipeline = Pipeline([
                        ('features', features),
                        ('clf', LogisticRegression(C=1.0, penalty='l1', solver='liblinear', dual=False))
                ])
    best_model = pipeline.fit(df_model.drop('polarity', axis=1), df_model.polarity)
    import pandas as pd
    new_positive_tweets = pd.Series(text)

    tc = text_count.TextCount()
    ct = clean_text.CleanText()
    df_counts_pos =tc.transform(new_positive_tweets)
    df_clean_pos = ct.transform(new_positive_tweets)
    df_model_pos = df_counts_pos
    df_model_pos['clean_text'] = df_clean_pos
    joblib.dump(best_model, 'ml_code/output/best_logreg_final_model.pkl')
    print("Predicting...")
    print(best_model.predict(df_model_pos).tolist())


if __name__ == '__main__':
    """
    This part is used for separate analysis and running of this code.
    sys.path.insert() is being used for running this module independently
    of other modules for testing purposes.
    
    This portion will only run when this module is specifically called.
    __name__ is equal to __main__ only for the module which is run, 
    not for the ones which are imported.
    """

    #########################################################################
    #                  REVIEW FILE DATA - IMDB ONLY                         #
    #########################################################################
    filename = "data_models/raw_data/imdb_labelled.txt"
    delimiter = r'\s{3,}'
    #########################################################################
    #             READ AND REINDEX TO AVOID DATA COLLECTION BIAS            #
    #########################################################################
    reindexed_data = read_and_reindex(filename=filename, delimiter=delimiter)
    #########################################################################
    #                          PRE-PROCESSING STEPS                         #
    #                            ANALYZE RAW DATA                           #
    #########################################################################
    # 1. visualize_target_class_frequency(reindexed_data)
    word_count_frame = clean_up_data(reindexed_data)
    # 3. visualize_word_count_and_polarity(word_count_frame)
    show_distribution(word_count_frame, 'count_words')
    word_count_frame = clean_up_data(reindexed_data)
    #########################################################################
    #                            CLEAN DATA                                 #
    #########################################################################
    cleaned_review = text_cleaner(reindexed_data)
    cleaned_review = fill_empty_reviews_with_no_text(cleaned_review=cleaned_review, filler_text="[no_review_here]")
    #########################################################################
    #                       WORD COUNT IN REVIEW                            #
    #########################################################################
    analyse_count_vectorizer_feature(cleaned_review)
    #########################################################################
    #                     CREATE TRAIN TEST DATA                            #
    #########################################################################
    X_train, X_test, y_train, y_test = create_test_data(word_count_frame, cleaned_review)

    #########################################################################
    #                    FIND CLASSIFIER AND MODEL                          #
    #            SET HYPER-PARAMETERS FOR THE CLASSIFIER                    #
    #                       GRID SEARCH THE MODEL                           #
    #             MULTINOMIAL NAIVE BAYES && LOGISTIC REGRESSION            #
    #########################################################################
    parameters_mnb, parameters_vect, parameters_logreg = seed_model_before_start()
    best_mnb_countvect = find_model_using_gridsearch(parameters_mnb=parameters_mnb,
                                                     parameters_vect=parameters_vect,
                                                     parameters_logreg=parameters_logreg,
                                                     data={'X_train': X_train, 'X_test': X_test})
    text = ["TThank you @VirginAmerica for you amazing customer support team on Tuesday 11/28 at @EWRairport and returning my lost bag in less than 24h! #efficiencyiskey #virginamerica",
            "Love flying with you guys ask these years. Sad that this will be the last trip 😂 @VirginAmerica #LuxuryTravel",
            "Wow @VirginAmerica main cabin select is the way to fly!! This plane is nice and clean & I have tons of legroom! Wahoo! NYC bound! ✈️",
            "This service is shit. I hate it."]
    predict_sentiment(text)
