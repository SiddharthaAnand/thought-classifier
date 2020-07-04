from main_app import db
from sqlalchemy.dialects.postgresql import JSON


class Result(db.Model):
    __tablename__ = 'user_raw_data'

    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.String())
    # will store polarity
    result = db.Column(JSON)
    result_without_stopwords = db.Column(JSON)

    def __init__(self, text=None, result=None, result_without_stopwords=None):
        self.text = text
        self.result = result
        self.result_without_stopwords = result_without_stopwords

    def __repr__(self):
        return '<id {}>'.format(self.id)