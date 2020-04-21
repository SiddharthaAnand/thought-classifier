from .data_file_reader import BaseFileReader
import pandas as pd


class TSVFileReader(BaseFileReader):
    def __init__(self, filename=None, separator=None):
        self.__init__("TSVReader", filename=filename, separator=separator)

    def read_file(self):
        pass

    def convert_to_df(self):
        return pd.read_csv(filepath_or_buffer=self.filename, sep=self.separator, lineterminator='\n')


def convert_tsv_to_data_frame(filename=None, separator=None):
    tsv_converter = TSVFileReader(filename=filename, separator=separator)
    return tsv_converter.convert_to_df()

