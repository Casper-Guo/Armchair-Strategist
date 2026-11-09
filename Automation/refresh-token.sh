#!/bin/bash

cd ~/Armchair-Strategist
{ read -r F1_USERNAME; read -r F1_PASSWORD; } < f1tv.txt
export F1_USERNAME F1_PASSWORD
python3 -m fastf1 auth f1tv --authenticate --method password
