class BaseFileReader(object):
    def __init__(self, reader="BaseFileReader"):
        self.name = reader

    def convert_to_df(self):
        """
        Needs to be overridden in every derived class.
        :return: A dataframe which consists of columns with |data | polarity | values
        """
        pass
