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


def read_and_reindex(filename=None, delimiter=None):
    if filename is not None and delimiter is not None:
        data = tsv_file_reader.convert_tsv_to_data_frame(filename=filename, delimiter=delimiter)

        ############################################
        #   Some basic information about data   ####
        ############################################
        print("Size of the dataset: %d " %(len(data['Text'])))

        # Shuffle the data randomly to avoid biasness while collection of the data
        data = data.reindex(np.random.permutation(data.index))
        data = data[['Text', 'Polarity']]
        return data


"""
Check for imbalance in the data set during the model training phase.
"""


def visualize_target_class_frequency(data=None):
    sns.catplot(x='Polarity', data=data, kind="count", height=5, aspect=1.5, palette='PuBuGn_d')
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
        df_eda = tc.fit_transform(data.Text)
        df_eda['Polarity'] = data.Polarity
        return df_eda


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
    # visualize_target_class_frequency(reindexed_data)
    word_count_frame = clean_up_data(reindexed_data)
    visualize_word_count_and_polarity(word_count_frame)