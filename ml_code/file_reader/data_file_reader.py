class BaseFileReader(object):
    def __init__(self, filename=None, separator=None):
        self.name = "base_file"
        self.filename = filename
        self.separator = separator
