#!/bin/bash
gunicorn main_app:app --daemon
python workers/worker.py