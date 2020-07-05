import sys
import random
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
    sentiment = ['Worse', 'Bad', 'an Average', 'Good', 'an Impressive', 'an Awesome']
    if request.method == 'POST':
        try:
            text = request.form['text']
            # TODO
            # Do your logic with the text
            # Send this to the model
            # Get result
            results['text'] = text.strip()
            results['sentiment'] = sentiment[random.randint(0, len(sentiment)-1)]
            result = Result(
                text=text,
                result=results,
                result_without_stopwords=results
            )
            # db.session.add(result)
            # db.session.commit()
        except Exception as e:
            print(e)
            errors.append(
                "Error encountered while inserting to database. Please try again.."
            )
    return render_template('home.html', errors=errors, results=results)


if __name__ == '__main__':
    app.config.from_object('development_config.DevelopmentConfig')
    app.run()
