import sys
from os import urandom, environ
from flask import Flask, render_template, flash
from flask_bootstrap import Bootstrap
from search_form import SearchForm
from flask_sqlalchemy import SQLAlchemy

sys.path.insert(0, 'configmodule/')
app = Flask(__name__)
app.config['SECRET_KEY'] = urandom(32)
app.config.from_object(environ['APP_SETTINGS'])
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
Bootstrap(app)


@app.route('/')
@app.route('/sentiment', methods=['POST', 'GET'])
def my_sentiment():
    thought_page = SearchForm()
    if thought_page.validate_on_submit():
        # Analzye code
        flash('Analyzing your thoughts...')
    return render_template('home.html', title='Understand your thoughts!', form=thought_page)


if __name__ == '__main__':
    app.config.from_object('development_config.DevelopmentConfig')
    app.run()
