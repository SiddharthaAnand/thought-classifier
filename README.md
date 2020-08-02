# Thought classifier
Find out the sentiment of your thoughts by simply typing here and clicking
a button.

[https://thought-classifier-staging.herokuapp.com/](https://thought-classifier-staging.herokuapp.com/)

## What it does?
It classifies your thoughts as positive (:)), negative (:() or neutral (:|).

## How does it do it?
We use a machine learning model which improves over time and tries
your group your thoughts as positive,  negative or neutral. You
can provide real time feedback which the model takes into account
and tries to learn and improve!

## How can I contribute?

#### Create a virtual environment first.
```
$ python3 -m venv venv
```
#### Installation
[Install postgres](https://www.postgresql.org/download/linux/ubuntu/)

Start the postgresql service.
```
$ service postgresql start
```
For further steps to create postgres db, follow the steps at the end.
Create dev and other environment databases in psql.

For every change that you want to show up in staging/production, you
need to do a git push to that environment before running heroku commands.

To add heroku remotes.
```
$ git remote add stage https://git.heroku.com/thought-classifier-staging.git
```
Similarly for production as well.

To remove a wrongly added remote.
```
$ git remote remove stage
```

For this repository, the staging remote name is 'stage' and
production remote name is 'heroku'.


Set up config for staging and production databases.
```
$ heroku config:set APP_SETTINGS=config.ProductionConfig --remote heroku
```
Run the same command above for setting the staging environment.

Check environment variables in staging environment.
```
$ heroku config --app thought-classifier-staging
```
Create a database_url config variable.
```
$ heroku addons:create heroku-postgresql:hobby-dev --app thought-classifier-staging
```
Check back with the same command above to see if the variable got created.
Commit and push changes to heroku staging server.
```
$ git push stage master
```
Run the migration on staging.
```
$ heroku run python manage.py db upgrade --app thought-classifier-staging
```
Notice how we only ran the upgrade, not the init or migrate commands like before. We already have our migration file set up and ready to go; we just need to apply it against the Heroku database.
Add postgres add on for production
```
$ heroku addons:create heroku-postgresql:hobby-dev --app thought-classifier-production
Creating heroku-postgresql:hobby-dev on ⬢ thought-classifier-production... free
Database has been created and is available
 ! This database is empty. If upgrading, you can transfer
 ! data from another database with pg:copy
Created postgresql-shallow-45263 as DATABASE_URL
Use heroku addons:docs heroku-postgresql to view documentation
```
#### Clone the repository.
```
$ git clone https://github.com/SiddharthaAnand/thought-classifier.git
```
Switch directory.
```
$ cd thought-classifier/
```
This will install all the requirements/packages required to run this application.
```
$ pip install -r requirements.txt
```
Try running the application.
```
python migration_code/manage.py runserver
```
If you see something like this, then congratulations the application is successfully
installed.
```
 * Serving Flask app "main_app" (lazy loading)
 * Environment: production
   WARNING: This is a development server. Do not use it in a production deployment.
   Use a production WSGI server instead.
 * Debug mode: on
 * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
 * Restarting with stat
 * Debugger is active!
 * Debugger PIN: 129-594-436
```
For working on machine learning models, you can switch to ml_code/ and start working.

## Dataset
We are using Rotten Tomatoes reviews dataset.

## Running the database commands

```
$ psql
```
To open the command prompt for postgresql.
To switch to the dev database.
```
$ \c textualreviews_dev;
sid=> \c textualreviews_dev;
You are now connected to database "textualreviews_dev" as user "sid".
```
To run commands to fetch data
```
$ textualreviews_dev=> select * from user_raw_data;
```
To insert a sample data from the terminal of psql.
```
$ insert into user_raw_data(text, result, result_without_stopwords) values('This is a sample text', '{"sample key": "sample value"}','{"sample key": "sample value"}');
```

## Installation of redis
```
$ wget http://download.redis.io/releases/redis-6.0.5.tar.gz
$ tar xzf redis-6.0.5.tar.gz
$ cd redis-6.0.5
$ make
```

Install redis addon on heroku in the staging/production. Make sure your
heroku account is verified(credit card details are added).
```
$ heroku addons:create redistogo:nano --app thought-classifier-staging
Creating redistogo:nano on ⬢ thought-classifier-staging... free
Created redistogo-tetrahedral-01369 as REDISTOGO_URL
Use heroku addons:docs redistogo to view documentation
```

Check if the config for redis url is set with the following command.
```
$ heroku config --app thought-classifier-staging
```

Once that is set, you need to setup the following files.
These steps are needed since I am using a free account, so I have access
to a single dyno. So, one process runs as a daemon. Ideally, it should
be running on different dynos.
- heroku.sh (Create)
To write the command for running the main_app in a daemon process.
Run the worker process.
- Procfile (Update)
Include heroku.sh.

To test the changes locally, you can run
```
$ heroku local
```
Go to the local url and see if the app is up and running.
If it is, then you are ready to push the changes to staging/production.


##### Troubleshooting for postgres

*Error 1*
```
$ psql
psql: could not connect to server: No such file or directory
        Is the server running locally and accepting
        connections on Unix domain socket "/var/run/postgresql/.s.PGSQL.5432"?
```

Try the following:
```
$ sudo systemctl start postgresql@11-main
```
This command should start the cluster.
You can check the status of the cluster by following command.
```
$ pg_lsclusters
11  main    5432 online postgres /var/lib/postgresql/11/main /var/log/postgresql/postgresql-11-main.log
```
Error 2
```
$ psql
psql: FATAL:  role "sid" does not exist
```
This means there is no user named 'sid' in postgres.
To simply login, just try the following:
```
sudo -u postgres -i
```
What this does is logs you in as 'postgres' user which is the default
user in postgres.
You need to create a new role and assign to it the same rights as the
one of postgres.
Try the following:
First login as 'postgres'
```
$ sudo -u postgres -i
$ createuser sid
$ psql
postgres=# alter user sid createdb;
postgres=# exit
$ sudo -u sid -i
# createdb
# psql
sid=>
```
The db and role are named as postgres. You need to create a role in postgres
which is the username and a db. Alter command gives your role the access
rights as postgres. Then createdb command creates a db with the same name
as username. When you type psql, the db prompt is finally open!
Error 3
Installation of psycopg2
[Try steps given.](https://www.psycopg.org/docs/install.html)

Error 4
```
File "/app/.heroku/python/lib/python3.6/os.py", line 669, in __getitem__
    raise KeyError(key) from None
KeyError: 'APP_SETTINGS'
```
This error comes because you have not set the environment variable
called 'APP_SETTINGS' for the staging/production env.
Run the command mentioned above and you should be ok.

Error 5
```
$ redis-6.0.5/src/redis-server
...
Connection refused
```
Check if you have redis-server running or not.

Error 6
If you face this error in the terminal running the worker process.
```
  File "<frozen importlib._bootstrap>", line 986, in _gcd_import
  File "<frozen importlib._bootstrap>", line 969, in _find_and_load
  File "<frozen importlib._bootstrap>", line 956, in _find_and_load_unlocked
ImportError: No module named 'main_app'
```
Then, add the following in the worker module at the top.
```
sys.path.insert(0, '/home/sid/github/thought-classifier/')
```
Error 7
If you are getting "Address already in use" error after running the following
command,
```
$ heroku local
```
Then, you need to kill the python/flask process which is running as a
daemon in the backend.
```
$ sudo netstat -nlp | grep 5000
$ kill <pid>
```
