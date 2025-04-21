#!/bin/bash

source /root/venv/bin/activate
python path.py
nohup /root/venv/bin/python /root/dev/trading2/tests/livetrendbar_test.py > output.log 2>&1