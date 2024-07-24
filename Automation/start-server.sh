#!/bin/bash

set -x

cd ~/Armchair-Strategist
source ./env/bin/activate
gunicorn app:server -b :8000 >/dev/null 2>./Automation/dash.log &
sleep 3
pgrep gunicorn && lsof -i :8000

