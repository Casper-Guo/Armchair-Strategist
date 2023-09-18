wsl.exe cd /mnt/d/Projects/F1-Visualization
wsl.exe chmod +x data-refresh.sh
wsl.exe chmod +x auto-push.exp
wsl.exe data-refresh.sh
start Notepad "ETL.log"
