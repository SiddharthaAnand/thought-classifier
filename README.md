# Thought classifier
Find out the sentiment of your thoughts by simply typing here and clicking
a button.

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
python main_app.py
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

#### mongodb installation
(The following commands works for Ubuntu 16.04).
For a different system, follow [these steps](https://docs.mongodb.com/manual/tutorial).
Import the public key used by the package management system.
```
$ wget -qO - https://www.mongodb.org/static/pgp/server-4.2.asc | sudo apt-key add -
OK
```
Create a list file for mongodb.
```
$ echo "deb [ arch=amd64,arm64 ] https://repo.mongodb.org/apt/ubuntu xenial/mongodb-org/4.2 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-4.2.list
```
Reload local package database.
```
$ sudo apt-get update
```
Install mongodb
```
$ sudo apt-get install -y mongodb-org
```
Start mongodb
```
$ sudo systemctl start mongod
```
For stopping mongodb
```
$ sudo systemctl stop mongod
```
Start using mongo
```
$ mongo
```
For further information regarding mongodb running on Ubuntu, [read on](https://docs.mongodb.com/manual/tutorial/install-mongodb-on-ubuntu/).
Start contributing :)

## Dataset
We are using Rotten Tomatoes reviews dataset.

##### Troubleshooting for postgres
Error 1
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