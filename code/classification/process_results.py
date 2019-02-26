import os
from os.path import join, dirname, abspath, pardir

BASE_DIR = abspath(join(dirname(__file__), pardir, pardir))
RESULTS_DIR = join(BASE_DIR, 'results')

def get_stats(dirname):

	STATS_FILE = join(dirname, 'stats')

	with open(STATS_FILE) as f:
		lines = f.readlines()
		f_mean = float(lines[1].strip().split()[1])
		prec_mean = float(lines[2].strip().split()[1])
		rec_mean = float(lines[3].strip().split()[1])
		f_std = float(lines[7].strip().split()[1])
		prec_std = float(lines[8].strip().split()[1])
		rec_std = float(lines[9].strip().split()[1])

	return prec_mean, prec_std, rec_mean, rec_std, f_mean, f_std

if __name__ == "__main__":

	prec_mean, prec_std, rec_mean, rec_std, f_mean, f_std = get_stats(RESULTS_DIR)