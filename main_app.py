import sys
from os import urandom
from flask import Flask, render_template, flash
from flask_bootstrap import Bootstrap
from configmodule import development_config
from configmodule import production_config
from search_form import SearchForm

sys.path.insert(0, 'configmodule/')
app = Flask(__name__)
app.config['SECRET_KEY'] = urandom(32)
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
