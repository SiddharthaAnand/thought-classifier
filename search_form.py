from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired


class SearchForm(FlaskForm):
    search_word = StringField('What are you thinking today?', validators=[DataRequired()],
                              default="I am thinking about ...")
    submit = SubmitField('How am I doing?')