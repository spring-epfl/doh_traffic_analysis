# Data collection 

We use Vagrant + VirtualBox to set up the data collection. 

In order to collect data, perform the following steps:

1. Install VirtualBox.
2. Install Vagrant.
3. Edit **Vagrantfile** to set your required number of VMs (Line 2). The current configuration creates 1 VM.
4.  Change the bootstrap script based on what you want to run (Line 5) -- **bootstrap.sh** for normal experiments, **bootstrap_tor.sh** for DNS-over-Tor experiments. The current configuration runs bootstrap.sh.
5. Create two directories **pcaps** and **logs** inside the vagrant folder.
6. Run the command `vagrant up` (should be run from inside the vagrant folder).
7. Note that sometimes, the cloudflared configuration copying throws up errors. Just run `vagrant reload --provision` after all the nodes are up and the error should disappear.
8. Log into the VM with the comman `vagrant ssh nodeN`, where N is the ID of the machine (N = 1, in the current configuration). 
9. Run the following command to kick off the experiment: `bash /vagrant/browse_chrome.sh N >> /vagrant/logs/$1/browse1\_$(date +%Y-%m-%d) 2>&1 &`, where N is the ID of the machine. 
10. Note: browse_chrome.sh uses Chrome (chrome_driver.py and chromedriver). If you need to use Firefox, please switch to firefox_driver.py and use the Firefox binary.

