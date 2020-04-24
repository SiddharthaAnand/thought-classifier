from .data_file_reader import BaseFileReader
import pandas as pd


class TSVFileReader(BaseFileReader):
    def __init__(self, filename=None, delimiter=None):
        super().__init__(reader="TSVReader")
        self.filename = filename
        self.delimiter = delimiter

    def read_file(self):
        pass

    def convert_to_df(self):
        return pd.read_csv(filepath_or_buffer=self.filename,
                           delimiter=self.delimiter,
                           lineterminator='\n',
                           names=['Text', 'Polarity'],
                           header=None,
                           na_values=[""],
                           engine='python')


def convert_tsv_to_data_frame(filename=None, delimiter=None):
    tsv_converter = TSVFileReader(filename=filename, delimiter=delimiter)
    return tsv_converter.convert_to_df()

