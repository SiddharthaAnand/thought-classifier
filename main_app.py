from flask import Flask

app = Flask(__name__)


@app.route('/')
def root_method():
    return 'Hello, Beautiful!'
