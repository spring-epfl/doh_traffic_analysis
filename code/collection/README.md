# Data collection for browsing experiments using Vagrant

Perform the following steps:

1. Install VirtualBox.
2. Install Vagrant.
3. Edit Vagrantfile to change the number of VMs. Change the bootstrap script based on what you want to run - bootstrap.sh for normal experiments, bootstrap_tor.sh for DNS-over-Tor experiments.
4. Edit bootstrap.sh to change the cron job timing.
5. Create two directories 'pcaps' and 'logs' inside the vagrant folder.
6. Run the command "vagrant up" (should be run from inside the vagrant folder).
7. Note that sometimes, the cloudflared configuration copying throws up errors. Just run "vagrant reload --provision" after all the nodes are up and the error should disappear.
