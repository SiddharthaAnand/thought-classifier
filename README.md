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