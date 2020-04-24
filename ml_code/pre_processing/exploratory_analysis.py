"""
Exploratory Analysis of the textual data available for sentiment analysis.

##      Analysis of Raw data | Number of positives/negatives polarity | Need for a balanced training data
##      Visualization
##      Split into training/testing data
##      k-fold cross validation
##      Generate a model
##      Pickle/serialize the model
"""


def analysis(filename=None, delimiter=None):
    if filename is not None and delimiter is not None:
        data = tsv_file_reader.convert_tsv_to_data_frame(filename=filename, delimiter=delimiter)
        print(data['Text'])
        # difference_between_dataframe_and_actual_file(filename, data_frame=data)


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
    filename = "data_models/raw_data/imdb_labelled.txt"
    delimiter = r'\s{3,}'
    analysis(filename=filename, delimiter=delimiter)
