#!/bin/bash
gunicorn app:main_app --daemon
python workers/worker.py