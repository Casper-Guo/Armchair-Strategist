#!/bin/bash

set -Eeuo pipefail
set -x

cd /mnt/d/Projects/F1-Visualization
exec > ETL.log 2>&1
date

{
    sleep 5m
    kill $$
} &

BRANCH=$(git branch --show-current)
PASSWORD=$(cat .password)

if [ $BRANCH != "main" ]
then
    git stash -u
    git checkout main
fi

python3 src/preprocess.py
python3 src/readme_update.py >/dev/null
git add .
git commit -m "Automatic data refresh" || true # ignore non-zero exit status when there's no diff on main 
./auto-push.exp "$PASSWORD" -d

if [ $BRANCH != "main" ]
then
    git checkout "$BRANCH"
    git stash pop || true # ignore non-zero exit status when there's no diff on branch
fi

echo "Success"