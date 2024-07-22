#!/bin/bash

if ! lsof -i :8000
then
    UTC=$(date)
    bash ~/Armchair-Strategist/Automation/start-server.sh
    aws sns publish --topic-arn arn:aws:sns:us-east-2:637423600104:Armchair-Strategist --message "WARNING: The server had been stopped in the last 30 minutes" --subject "Server Health Warning - $UTC"
fi
