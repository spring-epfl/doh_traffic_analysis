import os
from os.path import join, dirname, abspath, pardir
import pandas as pd
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

from pipeline.ngrams_classif1 import NgramsExtractor
from pipeline.tsfresh_basic import TSFreshBasicExtractor

from utils.util import *
from classify_pipeline import *
import time

BASE_DIR = abspath(join(dirname(__file__), pardir, pardir))

#Input data locations -- we can use either the data directory or the pickle file (if it exists)
DATA_DIR_CLOSED = join(BASE_DIR, 'dataset', 'loc1')
PICKLE_PATH_CLOSED = join(BASE_DIR, 'dataset', 'index.pickle')
DATA_DIR_OPEN = join(BASE_DIR, 'dataset', 'ow')
PICKLE_PATH_OPEN = join(BASE_DIR, 'dataset', 'indexow.pickle')

#Output files
RESULTS_DIR = join(BASE_DIR, 'results') #all results will be placed in the results directory
OUTPUT_STATISTICS = join(RESULTS_DIR, 'ow_stats') #precision/recall/f-score stats (mean and std)
OUTPUT_REPORT = join(RESULTS_DIR, 'ow_report') #detailed report (precision/recall/f-score for each class in each fold)
OUTPUT_TP = join(RESULTS_DIR, 'ow_tp_') #true vs predicted label (for other analysis if required)
OUTPUT_ACC = join(RESULTS_DIR, 'ow_acc') #accuracy for each fold
OUTPUT_PROB = join(RESULTS_DIR, 'ow_prob_') #predicition probability for each class along with truth label

def get_subsets(df, folds=10):
	""" Create training and test data subsets for k-fold cross validation. """
	kf = StratifiedKFold(n_splits=folds)
	train_sets = []
	test_sets = []
	for k, (train, test) in enumerate(kf.split(df, df.class_label)):
		print "Fold", k
		train_sets.append(df.iloc[train])
		test_sets.append(df.iloc[test])
	return train_sets, test_sets

def get_training_data(df_monitored_closed, df_unmonitored_closed, num_samples_monitored_closed, num_classes_unmonitored_closed, num_classes_monitored_closed):
	""" Function to create training dataset. 
	    We want training to be balanced - so equal number of total samples from each dataset - monitored-closed and unmonitored-closed.
	    Example: We have monitored-closed dataset of 50 classes with 36 samples each = 1800 samples in total.
	    We have 450 classes in unmonitored closed. We set number of samples to be 4 so that total samples = 1800."""
	num_samples_unmonitored_closed = (num_samples_monitored_closed * num_classes_monitored_closed) / num_classes_unmonitored_closed
	df_unmonitored_closed_trimmed = trim_sample_df(df_unmonitored_closed, num_samples_unmonitored_closed, map(str, range(num_classes_monitored_closed, num_classes_monitored_closed + num_classes_unmonitored_closed)))
	print "Monitored closed training shape:", "Samples", num_samples_monitored_closed, "Classes", num_classes_monitored_closed, "Shape", df_monitored_closed.shape
	print "Unmonitored closed training shape:", "Samples", num_samples_unmonitored_closed, "Classes", num_classes_unmonitored_closed, "Shape", df_unmonitored_closed_trimmed.shape
	return pd.concat([df_monitored_closed, df_unmonitored_closed_trimmed])

def ow_experiment():
	""" Function to run open world experiment. Creates monitored and unmonitored datasets from the LOC1 and OW datasets.	"""

	if not os.path.isdir(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
	
	#Load closed and open world data
	if os.path.isfile(PICKLE_PATH_CLOSED):
		df_closed = load_data(path=PICKLE_PATH_CLOSED)
	else:
		df_closed = load_data(path=DATA_DIR_CLOSED)
	if os.path.isfile(PICKLE_PATH_OPEN):
		df_open = load_data(path=PICKLE_PATH_OPEN)
	else:
		df_open = load_data(path=DATA_DIR_OPEN)

	#Monitored class is 1% of whole dataset. Closed world is 10% of whole dataset.
	num_classes_monitored_closed = 50
	num_classes_unmonitored_closed = 450
	num_classes_closed = 500
	num_classes_unmonitored_open = 4500

	#Number of samples we set. 
	num_samples_test = 3
	num_samples_test_closed = 2
	num_samples_training = 36
	num_samples_monitored_closed = 36
	num_samples_closed = 40

	#Trim closed world dataset
	df_closed_trimmed = trim_sample_df(df_closed, num_samples_closed, map(str, range(num_classes_monitored_closed + num_classes_unmonitored_closed)))
	print "Closed world shape:", "Samples", num_samples_closed, "Classes", num_classes_closed, "Shape", df_closed_trimmed.shape
	
	#Get subsets for training by k-fold of closed world
	df_closed_trimmed = df_closed
	train_sets_closed, test_sets_closed = get_subsets(df_closed_trimmed)
	
	#Get open-unmonitored and split into subsets
	print "Open world shape:", "Samples:", num_samples_test, df_open.shape
	df_unmonitored_open = trim_sample_df(df_open, num_samples_test, map(str, range(1500, 1500 + num_classes_unmonitored_open)))
	print "Unmonitored open shape:", df_unmonitored_open.shape
	test_sets_open = np.array_split(df_unmonitored_open, 10)
	results = []

	for i in range(0, len(train_sets_closed)):
		print "Iteration:", i
		#Split closed world into monitored and un-monitored
		df_closed = train_sets_closed[i]
		print "Initial training subset shape:", df_closed.shape
		df_closed_trimmed = trim_sample_df(df_closed, num_samples_training, map(str, range(num_classes_monitored_closed + num_classes_unmonitored_closed)))
		print "Trimming", df_closed_trimmed.shape
		df_monitored_closed = df_closed_trimmed[df_closed_trimmed["class_label"].astype(int) < 50]
		df_unmonitored_closed = df_closed_trimmed[df_closed_trimmed["class_label"].astype(int) >= 50]
		print "Monitored closed shape:", "Samples", num_samples_training, "Classes", num_classes_monitored_closed, "Shape", df_monitored_closed.shape
		print "Unmonitored closed shape:", "Samples", num_samples_training, "Classes", num_classes_unmonitored_closed, "Shape", df_unmonitored_closed.shape
		#Create training dataset
		df_train = get_training_data(df_monitored_closed, df_unmonitored_closed, num_samples_monitored_closed, num_classes_unmonitored_closed, num_classes_monitored_closed)
		print "Training data shape:", df_train.shape
		#Create test dataset out of closed + open
		df_closed_test_trimmed = trim_sample_df(test_sets_closed[i], num_samples_test_closed, map(str, range(num_classes_monitored_closed + num_classes_unmonitored_closed)))
		df_test = pd.concat([df_closed_test_trimmed, test_sets_open[i][:1000]])
		print "Test data shape:", df_test.shape
		results.append(classify(df_train, df_test, OUTPUT_ACC, OUTPUT_PROB + str(i)))
	
	#Obtain classification reports
	report = describe_classif_reports(results, OUTPUT_TP)
	print_stats_paper(report, OUTPUT_STATISTICS)
	with open(OUTPUT_REPORT, "w") as f:
		f.write(report.to_string())
	stop = time.time()	

if __name__ == '__main__':
	
	ow_experiment()

