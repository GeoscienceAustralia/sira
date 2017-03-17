#!/bin/bash
export PYTHONPATH="${PYTHONPATH+$PYTHONPATH:}"$(pwd)
export PYTHONPATH=$PYTHONPATH:$(dirname $(pwd))
export FLASK_APP=web.py
export FLASK_DEBUG=1
flask run -h 0.0.0.0

