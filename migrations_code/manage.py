# Migration
import sys
sys.path.insert(0, '/home/sid/github/thought-classifier/')

import os
from flask_script import Manager
from flask_migrate import Migrate, MigrateCommand

from main_app import app, db

MIGRATION_DIR = os.path.join('migrations_code', 'migrations')
app.config.from_object(os.environ['APP_SETTINGS'])

migrate = Migrate(app, db, directory=MIGRATION_DIR)
manager = Manager(app)

manager.add_command('db', MigrateCommand)


if __name__ == '__main__':
    manager.run()