#!/usr/bin/env bash

sudo apt-get update
sudo apt-get install -y python-pip xvfb libnss3-dev chromium-browser

sudo pip install selenium
sudo pip install pyvirtualdisplay

sudo apt install /vagrant/cloudflared-stable-linux-amd64.deb
sudo mkdir -p /usr/local/etc/cloudflared
sudo chmod 777 /usr/local/etc/cloudflared/
sudo cat << EOF > /usr/local/etc/cloudflared/config.yaml
proxy-dns: true
proxy-dns-upstream:
 - https://1.1.1.1/dns-query
 - https://1.0.0.1/dns-query
EOF

sudo cp /usr/local/etc/cloudflared/config.yaml /etc/cloudflared/config.yml
sudo cp /etc/cloudflared/cert.pem /usr/local/etc/cloudflared/

sudo cloudflared service install

sudo cp /vagrant/chromedriver /usr/local/bin/
sudo cp /vagrant/resolv.conf /etc/

sudo timedatectl set-timezone Europe/Zurich

mkdir /vagrant/pcaps/$1
mkdir /vagrant/pcaps2/$1
mkdir /vagrant/logs/$1

# cron="00 20 * * * bash /vagrant/browse_chrome.sh $1 >> /vagrant/logs/$1/browse1\_$(date +%Y-%m-%d) 2>&1
# 00 08 * * * bash /vagrant/browse_chrome2.sh $1 >> /vagrant/logs/$1/browse2\_$(date +%Y-%m-%d) 2>&1"

# # Escape all the asterisks so we can grep for it
# cron_escaped=$(echo "$cron" | sed s/\*/\\\\*/g)

# # Check if cron job already in crontab
# crontab -l | grep "${cron_escaped}"
# if [[ $? -eq 0 ]] ;
#   then
#     echo "Crontab already exists. Exiting..."
#     exit
#   else
#     # Write out current crontab into temp file
#     crontab -l > mycron
#     # Append new cron into cron file
#     echo "$cron" > mycron
#     # Install new cron file
#     crontab mycron
#     # Remove temp file
#     rm mycron
# fi

