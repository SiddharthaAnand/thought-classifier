import sys
from os import urandom, environ
from flask import Flask, render_template, request
from flask_bootstrap import Bootstrap
from flask_sqlalchemy import SQLAlchemy

sys.path.insert(0, 'configmodule/')
app = Flask(__name__)
app.config['SECRET_KEY'] = urandom(32)
app.config.from_object(environ['APP_SETTINGS'])
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

from data_models.models import Result

Bootstrap(app)


@app.route('/', methods=['POST', 'GET'])
def my_sentiment():
    errors = []
    results = {}
    if request.method == 'POST':
        try:
            text = request.form['text']
            # Do your logic with the text
        except:
            errors.append(
                "Error encountered while processing. Please try again.."
            )
    return render_template('home.html', errors=errors, results=results)


if __name__ == '__main__':
    app.config.from_object('development_config.DevelopmentConfig')
    app.run()
