#!/bin/bash

set -Eeuo pipefail
set -x

cd ~/Armchair-Strategist
exec > ./Automation/data-refresh.log 2>&1

handle_failure() {
    error_line=$BASH_LINENO
    error_command=$BASH_COMMAND

    if [[ "$error_command" == *preprocess.py* ]]
    then
        # failure in preprocessing, bad data might have been written to file
        git restore .
    elif [[ "$error_command" == *readme_machine.py* ]]
    then
        # failure in making README graphics, withhold all graph updates only
        git restore Docs/visuals/*
        git add .
        git commit -m "Partial data refresh (no visualizations)" || true # ignore non-zero exit status when there's no diff on main
        ./Automation/auto-push.exp -d 2>./Automation/auto-push.log
    fi

    # relaunch server
    ./Automation/start-server.sh

    aws sns publish --topic-arn arn:aws:sns:us-east-2:637423600104:Armchair-Strategist --message file://./Automation/data-refresh.log --subject "Data Refresh Failure - $error_line: $error_command"
}
trap handle_failure ERR
trap handle_failure SIGTERM

source ./env/bin/activate 2>/dev/null
UTC=$(date)
# shutdown dash app, ignore non-zero return status in case there is no gunicorn process running
pkill -cef gunicorn || :

python3 f1_visualization/preprocess.py
python3 f1_visualization/readme_machine.py --update-readme >/dev/null
git add .
git commit -m "Automatic data refresh" || true # ignore non-zero exit status when there's no diff on main
./Automation/auto-push.exp -d 2>./Automation/auto-push.log

# relaunch dash app
./Automation/start-server.sh
aws sns publish --topic-arn arn:aws:sns:us-east-2:637423600104:Armchair-Strategist --message file://./Automation/data-refresh.log --subject "Data Refresh Success - $UTC"
