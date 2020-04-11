import sys
from flask import Flask
from configmodule import development_config
from configmodule import production_config

sys.path.insert(0, 'configmodule/')
app = Flask(__name__)


@app.route('/')
def root_method():
    return 'Hello, Beautiful!'


if __name__ == '__main__':
    app.config.from_object('production_config.ProductionConfig')
    app.run()
