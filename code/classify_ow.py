import os
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

def get_subsets(df, folds=10):
	kf = StratifiedKFold(n_splits=folds)
	train_sets = []
	test_sets = []
	for k, (train, test) in enumerate(kf.split(df, df.class_label)):
		print "Fold", k
		train_sets.append(df.iloc[train])
		test_sets.append(df.iloc[test])
	return train_sets, test_sets

def get_training_data(df_monitored_closed, df_unmonitored_closed, num_samples_monitored_closed, num_classes_unmonitored_closed, num_classes_monitored_closed):

	#We want training to be balanced - so equal number of total samples from each dataset - monitored-closed and unmonitored-closed.
	#Example: We have monitored-closed dataset of 50 classes with 36 samples each = 1800 samples in total.
	#We have 450 classes in unmonitored closed. We set number of samples to be 4 so that total samples = 1800.
	num_samples_unmonitored_closed = (num_samples_monitored_closed * num_classes_monitored_closed) / num_classes_unmonitored_closed
	df_unmonitored_closed_trimmed = trim_sample_df(df_unmonitored_closed, num_samples_unmonitored_closed, map(str, range(num_classes_monitored_closed, num_classes_monitored_closed + num_classes_unmonitored_closed)))
	print "Monitored closed training shape", "Samples", num_samples_monitored_closed, "Classes", num_classes_monitored_closed, df_monitored_closed.shape
	print "Unmonitored closed training shape", "Samples", num_samples_unmonitored_closed, "Classes", num_classes_unmonitored_closed, df_unmonitored_closed_trimmed.shape
	return pd.concat([df_monitored_closed, df_unmonitored_closed_trimmed])

def ow_experiment():

	data_dir_closed = '../../experiments/processed_traces/vagrant/'
	pickle_path_closed = 'index.pickle'
	data_dir_open = '../../experiments/processed_traces/openworld/'
	pickle_path_open = 'indexow.pickle'
	output_statistics = "results/openworld2/stats" #precision/recall/f-score stats
	output_report = "results/openworld2/report" #detailed report
	output_tp = "results/openworld2/tp_" #true vs predicted label (for other analysis if required)
	output_acc = "results/openworld2/acc" #accuracy for each fold
	output_prob = "results/openworld2/prob_"
	
	#Load closed and open world data
	if os.path.isfile(pickle_path_closed):
		df_closed = load_data(path=pickle_path_closed)
	else:
		df_closed = load_data(path=data_dir_closed)
	if os.path.isfile(pickle_path_open):
		df_open = load_data(path=pickle_path_open)
	else:
		df_open = load_data(path=data_dir_open)

	#opt_num_insts, opt_num_classes = optimal_instances_per_class(df_open, draw=True)
	#print "optimal", opt_num_insts, opt_num_classes

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

	df_closed_trimmed = trim_sample_df(df_closed, num_samples_closed, map(str, range(num_classes_monitored_closed + num_classes_unmonitored_closed)))
	print "Closed world shape:", "Samples", num_samples_closed, "Classes", num_classes_closed, df_closed_trimmed.shape
	
	#Get subsets for training by k-fold of closed world
	df_closed_trimmed = df_closed
	train_sets_closed, test_sets_closed = get_subsets(df_closed_trimmed)
	
	#Get open-unmonitored and split into subsets
	print "Open world shape:", "Samples", num_samples_test, df_open.shape
	print df_open.class_label.min(), df_open.class_label.max()
	df_unmonitored_open = trim_sample_df(df_open, num_samples_test, map(str, range(1500, 1500 + num_classes_unmonitored_open)))
	print "Unmonitored open shape:", df_unmonitored_open.shape
	test_sets_open = np.array_split(df_unmonitored_open, 10)
	results = []

	for i in range(0, len(train_sets_closed)):
		#print i
		#Split closed world into monitored and un-monitored
		df_closed = train_sets_closed[i]
		print "Initial training subset shape:", df_closed.shape
		df_closed_trimmed = trim_sample_df(df_closed, num_samples_training, map(str, range(num_classes_monitored_closed + num_classes_unmonitored_closed)))
		print "Trimming", df_closed_trimmed.shape
		df_monitored_closed = df_closed_trimmed[df_closed_trimmed["class_label"].astype(int) < 50]
		df_unmonitored_closed = df_closed_trimmed[df_closed_trimmed["class_label"].astype(int) >= 50]
		print "Monitored closed shape:", "Samples", num_samples_training, "Classes", num_classes_monitored_closed, df_monitored_closed.shape
		print "Unmonitored closed shape:", "Samples", num_samples_training, "Classes", num_classes_unmonitored_closed, df_unmonitored_closed.shape
		#Create training dataset
		df_train = get_training_data(df_monitored_closed, df_unmonitored_closed, num_samples_monitored_closed, num_classes_unmonitored_closed, num_classes_monitored_closed)
		print "Training data shape:", df_train.shape
		#Create test dataset out of closed + open
		df_closed_test_trimmed = trim_sample_df(test_sets_closed[i], num_samples_test_closed, map(str, range(num_classes_monitored_closed + num_classes_unmonitored_closed)))
		df_test = pd.concat([df_closed_test_trimmed, test_sets_open[i][:1000]])
		print "Test closed shape:", df_closed_test_trimmed.shape, test_sets_open[i].shape
		print "Test data shape:", df_test.shape
		results.append(classify(df_train, df_test, output_acc, output_prob + str(i)))
		#break
	
	report = describe_classif_reports(results, output_tp)
	print_stats_paper(report, output_statistics)
	with open(output_report, "w") as f:
		f.write(report.to_string())
	stop = time.time()	

if __name__ == '__main__':
	
	ow_experiment()

