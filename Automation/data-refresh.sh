#!/bin/bash

set -Eeuo pipefail
set -x

cd ~/Armchair-Strategist
exec > ./Automation/data-refresh.log 2>&1

handle_failure() {
    error_line=$BASH_LINENO
    error_command=$BASH_COMMAND

    aws sns publish --topic-arn arn:aws:sns:us-east-2:637423600104:Armchair-Strategist --message file://./Automation/data-refresh.log --subject "Data Refresh Failure - $error_line: $error_command"
}
trap handle_failure ERR
trap handle_failure SIGTERM

date
UTC=$(date)
# shutdown dash app, ignore non-zero return status in case there is no gunicorn process running
pkill -f gunicorn || :

python3 f1_visualization/preprocess.py
python3 f1_visualization/readme_machine.py --update-readme >/dev/null
git add .
git commit -m "Automatic data refresh" || true # ignore non-zero exit status when there's no diff on main
./Automation/auto-push.exp -d

# relaunch dash app
gunicorn app:server -b :8000 >/dev/null 2>./Automation/dash.log &
pgrep gunicorn
aws sns publish --topic-arn arn:aws:sns:us-east-2:637423600104:Armchair-Strategist --message file://./Automation/data-refresh.log --subject "Data Refresh Success - $UTC"
