#!/bin/bash
set -Eeuo pipefail
set -x

cd ~/F1-Data-Visualization
exec > ./Automation/sync-code.log 2>&1
date

{
    sleep 1m
    kill $$
} &

# assume EC2 instance running from main
# shutdown dash app
pkill -f gunicorn

./Automation/auto-pull.exp -d

# relaunch dash app
gunicorn app:server -b :8000
aws sns publish --topic-arn arn:aws:sns:us-east-2:637423600104:Armchair-Strategist --message file://./Automation/sync-code.log --subject "Code Syncing Success"
echo "Success"

# If terminated midway, send log to email
|| aws sns publish --topic-arn arn:aws:sns:us-east-2:637423600104:Armchair-Strategist --message file://./Automation/sync-code.log --subject "Code Syncing Failure"
