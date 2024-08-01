#!/bin/bash
set -Eeuo pipefail
set -x

cd ~/Armchair-Strategist
exec > ./Automation/start-server.log 2>&1

handle_failure() {
    error_line=$BASH_LINENO
    error_command=$BASH_COMMAND

    aws sns publish --topic-arn arn:aws:sns:us-east-2:637423600104:Armchair-Strategist --message file://./Automation/start-server.log --subject "Start Server Failure - $error_line: $error_command"
}
trap handle_failure ERR

source ./env/bin/activate 2>/dev/null
gunicorn app:server -b :8000 >/dev/null 2>./Automation/dash.log &
sleep 3
pgrep gunicorn && lsof -i :8000

