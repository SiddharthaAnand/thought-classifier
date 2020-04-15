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
> This is a Work In Progress.

Clone the repository.
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

## mongo installation
```
$ wget -qO - https://www.mongodb.org/static/pgp/server-4.2.asc | sudo apt-key add -
OK
```

Start contributing :)