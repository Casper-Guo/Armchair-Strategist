source ~/F1-Data-Visualization/env/bin/activate
sudo systemctl start nginx
gunicorn app:server -b :8000 &
