import time
from selenium import webdriver
from selenium.common.exceptions import TimeoutException
import subprocess
from pyvirtualdisplay import Display
import sys
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.firefox.firefox_binary import FirefoxBinary

start = time.time()
INTERVAL_TIME = 1 #Interval time between queries
urls = []
ct = 0

fname = "/vagrant/short_list_1500"
with open(fname) as f:
	lines = f.readlines()
	for line in lines:
		urls.append(line.strip())

url = urls[int(sys.argv[1])]
print(url)
display = Display(visible=0, size=(800, 800))
display.start()
print("Started display")

#Change path of binary
binary = FirefoxBinary('firefox/firefox')
#Change resolver URL as required
resolver_url = 'https://mozilla.cloudflare-dns.com/dns-query'
#resolver_url = 'https://dns.google.com/experimental'

fp = webdriver.FirefoxProfile()
fp.DEFAULT_PREFERENCES['frozen']["network.trr.mode"] = 2
fp.DEFAULT_PREFERENCES['frozen']["network.trr.uri"] = resolver_url
driver = webdriver.Firefox(executable_path='/usr/local/bin/geckodriver', firefox_binary=binary, firefox_profile=fp)
print("Started Firefox driver")

url = 'http://' + url
try:
	driver.get(url)
	time.sleep(INTERVAL_TIME*5)
except TimeoutException as ex:
	print(ex)
driver.quit()
display.stop()
stop = time.time()
print("Time taken:" + str(stop - start))
time.sleep(INTERVAL_TIME)
