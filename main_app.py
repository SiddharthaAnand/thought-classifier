import sys
import json
import random
from rq import Queue
from rq.job import Job
from workers.worker import conn
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

q = Queue(connection=conn)
from data_models.models import Result

Bootstrap(app)


@app.route('/', methods=['POST', 'GET'])
def index():
    return render_template('home.html')


@app.route('/sentiment', methods=['POST'])
def analyze_sentiment():
    if request.method == 'POST':
        user_entered_text = json.loads(request.data.decode())['text']
        print('Entering method: {}'.format(user_entered_text))
        job = q.enqueue(
            func=get_sentiment,
            args=(user_entered_text,),
            result_ttl=5000
        )
        return job.get_id()


@app.route('/sentiment/<job_id>', methods=['GET'])
def get_results(job_id):

    job = Job.fetch(job_id, connection=conn)

    if job.is_finished:
        return str(job.result), 200
    else:
        return "Nay!", 202


def get_sentiment(text):
    sentiment = ['Worse', 'Bad', 'an Average', 'Good', 'an Impressive', 'an Awesome']
    # TODO
    # Do your logic with the text
    # Send this to the model
    # Get result
    results = {
        'text': text.strip(),
        'sentiment': sentiment[random.randint(0, len(sentiment) - 1)]
    }
    try:
        result = Result(
            text=text,
            result=results,
            result_without_stopwords=results
        )
        # db.session.add(result)
        # db.session.commit()
    except:
        results['errors'] = "Unable to add data to the database"
    return results


if __name__ == '__main__':
    app.config.from_object('development_config.DevelopmentConfig')
    app.run()
