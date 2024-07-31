#!/bin/bash
set -Eeuo pipefail
set -x

cd ~/Armchair-Strategist
exec > ./Automation/sync-code.log 2>&1

handle_failure() {
    error_line=$BASH_LINENO
    error_command=$BASH_COMMAND

    # relaunch server even if auto-pull fails
    ./Automation/start-server.sh

    aws sns publish --topic-arn arn:aws:sns:us-east-2:637423600104:Armchair-Strategist --message file://./Automation/sync-code.log --subject "Code Syncing Failure - $error_line: $error_command"
}
trap handle_failure ERR
trap handle_failure SIGTERM

source ./env/bin/activate 2>/dev/null
UTC=$(date)
# assume EC2 instance running from main
# shutdown dash app, ignore non-zero return status in case there is no gunicorn process running
pkill -cef gunicorn || :

./Automation/auto-pull.exp -d 2>./Automation/auto-pull.log

# relaunch dash app
./Automation/start-server.sh
aws sns publish --topic-arn arn:aws:sns:us-east-2:637423600104:Armchair-Strategist --message file://./Automation/sync-code.log --subject "Code Syncing Success - $UTC"
