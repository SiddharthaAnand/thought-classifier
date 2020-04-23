from .data_file_reader import BaseFileReader
import pandas as pd


class TSVFileReader(BaseFileReader):
    def __init__(self, filename=None, separator=None):
        super().__init__(reader="TSVReader")
        self.filename = filename
        self.separator = separator

    def read_file(self):
        pass

    def convert_to_df(self):
        return pd.read_csv(filepath_or_buffer=self.filename,
                           sep=self.separator,
                           lineterminator='\n',
                           names=['Text', 'Polarity'],
                           header=None,
                           na_values=[""])


def convert_tsv_to_data_frame(filename=None, separator=None):
    tsv_converter = TSVFileReader(filename=filename, separator=separator)
    return tsv_converter.convert_to_df()

