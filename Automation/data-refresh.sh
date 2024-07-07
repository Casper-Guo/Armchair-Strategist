#!/bin/bash

set -Eeuo pipefail
set -x

cd ~/F1-Data-Visualization
exec > ./Automation/data-refresh.log 2>&1
date

{
    sleep 5m
    kill $$
} &

# shutdown dash app
pkill -f gunicorn

python3 f1_visualization/preprocess.py
python3 f1_visualization/readme_machine.py --update-readme >/dev/null
git add .
git commit -m "Automatic data refresh" || true # ignore non-zero exit status when there's no diff on main 
./Automation/auto-push.exp -d

# relaunch dash app
gunicorn app:server -b :8000
aws sns publish --topic-arn arn:aws:sns:us-east-2:637423600104:Armchair-Strategist --message file://./Automation/data-refresh.log --subject "Data Refresh Success"
echo "Success"

# If terminated midway, send log to email
|| aws sns publish --topic-arn arn:aws:sns:us-east-2:637423600104:Armchair-Strategist --message file://./Automation/data-refresh.log --subject "Data Refresh Failure"
