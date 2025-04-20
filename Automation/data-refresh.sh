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
        aws sns publish --topic-arn arn:aws:sns:us-east-2:637423600104:Armchair-Strategist --message file://./Automation/data-refresh.log --subject "Data Refresh Failure - preprocess"
    elif [[ "$error_command" == *readme_machine.py* ]]
    then
        # failure in making README graphics, withhold all graph updates only
        git restore Docs/visuals/*
        git add .
        git commit -m "Partial data refresh (no visualizations)" || true # ignore non-zero exit status when there's no diff on main
        ./Automation/auto-push.exp -d 2>./Automation/auto-push.log
        aws sns publish --topic-arn arn:aws:sns:us-east-2:637423600104:Armchair-Strategist --message file://./Automation/data-refresh.log --subject "Data Refresh Failure - readme_machine"
    elif [[ "$error_command" == *reddit_machine.py* ]]
    then
        # failure in Reddit publishing, release all other data and emit warning
        ./Automation/auto-push.exp -d 2>./Automation/auto-push.log
        aws sns publish --topic-arn arn:aws:sns:us-east-2:637423600104:Armchair-Strategist --message file://./Automation/data-refresh.log --subject "Reddit Publication Warning"
    else
        aws sns publish --topic-arn arn:aws:sns:us-east-2:637423600104:Armchair-Strategist --message file://./Automation/data-refresh.log --subject "Data Refresh Failure - $error_line: $error_command"
    fi

    # relaunch server
    ./Automation/start-server.sh
}
trap handle_failure ERR
trap handle_failure SIGTERM

if [ $# -eq 1 ]; then
  if [ "$1" != "-g" ] && [ "$1" != "-s" ]; then
    echo "Error: Invalid flag for readme_machine.py."
    echo "Error: Use -g for grand prix and -s for sprint."
    exit 1
  fi
  flag="$1"
else
  flag="-g"
fi

source ./env/bin/activate 2>/dev/null
UTC=$(date)
# shutdown dash app, ignore non-zero return status in case there is no gunicorn process running
pkill -cef gunicorn || true

python3 f1_visualization/preprocess.py

# update README and commit if there are unstaged changes
if [[ -n "$(git status -s)" ]]; then
  python3 readme_machine.py --update-readme --reddit-machine "$flag" >/dev/null
  git add .
  git commit -m "Automatic data refresh"

  # post to Reddit when there is new graphics available
  python3 reddit_machine.py
fi

./Automation/auto-push.exp -d 2>./Automation/auto-push.log

# relaunch dash app
./Automation/start-server.sh
aws sns publish --topic-arn arn:aws:sns:us-east-2:637423600104:Armchair-Strategist --message file://./Automation/data-refresh.log --subject "Data Refresh Success - $UTC"
