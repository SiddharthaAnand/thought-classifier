#!/bin/bash
gunicorn app:app --daemon
python workers/worker.py