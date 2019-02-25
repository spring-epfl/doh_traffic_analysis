import sys
import os
from os.path import join, dirname, abspath, pardir
import pandas as pd

from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

from pipeline.ngrams_classif import NgramsExtractor
from pipeline.tsfresh_basic import TSFreshBasicExtractor

from utils.util import *
import time


BASE_DIR = abspath(join(dirname(__file__), pardir, pardir))
DATA_DIR = join(BASE_DIR, 'dataset')
CODE_DIR = join(BASE_DIR, 'code')
COLLEC_DIR = join(CODE_DIR, 'collection')
CLASSIF_DIR = join(CODE_DIR, 'classification')
RESULTS_DIR = join(BASE_DIR, 'results')

STRANGE_URLS_LIST = join(DATA_DIR, 'strange_urls', 'strange_urls_shortlist')
ALL_URL_LIST = join(COLLEC_DIR, 'short_list_500')

def print_stats_paper(report,  output, avg='macro avg', stats=['mean', 'std']):
    """Function to print overall statistics. Mean/Std of Precision/Recall/F-Score"""
    by_label = report.groupby('label').describe()
    with open(output, "w") as f:
        for stat in stats:
            print >>f, "Statistic:", stat
            print >>f, by_label.loc[avg].xs(stat, level=1)


def report_true_pred(y_true, y_pred, i, tag):
    """Function to print true and predicted labels."""
    tag += str(i)
    with open(tag, "w") as f:
        for i in range(0, len(y_true)):
            print >>f, y_true[i], y_pred[i]


def describe_classif_reports(results, tag):
    """Function to obtain detailed classification report."""
    true_vectors, pred_vectors = [r[0] for r in results], [r[1] for r in results]
    all_folds = pd.DataFrame(columns=['label', 'fold', 'precision', 'recall', 'f1-score', 'support'])
    for i, (y_true, y_pred) in enumerate(zip(true_vectors, pred_vectors)):
        report_true_pred(y_true, y_pred, i, tag)
        output = classification_report(y_true, y_pred, output_dict=True)
        df = pd.DataFrame(output).transpose().reset_index().rename(columns={'index': 'label'})
        df['fold'] = i
        all_folds = all_folds.append(df)
    return all_folds


def cross_validate(df, output_acc, folds=10):
    """Function to perform k-fold cross classification."""
    kf = StratifiedKFold(n_splits=folds)
    results = []
    for k, (train, test) in enumerate(kf.split(df, df.class_label)):
        print "Fold", k
        result = classify(df.iloc[train], df.iloc[test], output_acc)
        results.append(result)
    return results


def cross_classify(df1, df2, output_acc1, output_acc2, folds=10):
    """Function to perform k-fold cross classification in both directions (for cross-classification experiments)."""
    df1_df2_accs, df2_df1_accs = [], []
    kf = StratifiedKFold(n_splits=folds)
    for k, ((df1_train, df1_test), (df2_train, df2_test)) in enumerate(zip(kf.split(df1, df1.class_label), kf.split(df2, df2.class_label))):
        print "Fold", k
        df1_df2_accs.append(classify(df1.iloc[df1_train], df2.iloc[df2_test], output_acc1, output_acc1 + "_" + str(k)))
        df2_df1_accs.append(classify(df2.iloc[df2_train], df1.iloc[df1_test], output_acc2, output_acc2 + "_" + str(k)))
    return df1_df2_accs, df2_df1_accs

def classify(train, test, output_acc, output_prob=""):
    """Function that runs the classification pipeline."""
    # Feature extraction methods. Add/delete as required.
    combinedFeatures = FeatureUnion([
       #('tsfresh', TSFreshBasicExtractor()),
      ('ngrams', NgramsExtractor(max_ngram_len=1)),
    ])

    # Pipeline. Feature extraction + classification
    pipeline = Pipeline([
      ('features', combinedFeatures),
      ('clf', RandomForestClassifier(n_estimators=100))
    ])

    # Training with pipeline
    pipeline.fit(train, train.class_label)
    # Prediction
    y_pred = pipeline.predict(test)

    #If we want to print probabilities (for OW experiment)
    if len(output_prob) > 0:
        y_pred_prob = pipeline.predict_proba(test)
        with open(output_prob, "w") as f:
          class_names = [x for x in pipeline.classes_]
          s = '|'.join(class_names)
          print >>f, "Truth", "|", s
          truth_labels = list(test.class_label)
          for i in range(0, len(y_pred_prob)):
              preds = [str(x) for x in y_pred_prob[i]]
              preds = '|'.join(preds)
              print >>f, truth_labels[i], "|", preds

    acc = accuracy_score(test.class_label, y_pred)
    print "Accuracy Score:", acc
    with open(output_acc, "a") as f:
        print >>f, "Accuracy Score:", acc

    return list(test.class_label), list(y_pred)


def remove_strange(df, shortlist, urls):
    """Function to perform data cleaning by removing strange URLs."""
    strange_urls = []
    with open(shortlist) as f:
        lines = f.readlines()
        strange_urls = [x.strip() for x in lines]

    strange_urls = [urls.index(x) for x in strange_urls]
    print len(strange_urls)
    df = df[~df["class_label"].astype(int).isin(strange_urls)]
    return df


def get_interval(df, start_date, end_date):
    mask = (df['date'] >= start_date) & (df['date'] <= end_date)
    df = df.loc[mask]
    return df


def get_url_list(url_list):
    """Function to read the list of URLS."""
    urls = []
    with open(url_list) as f:
        lines = f.readlines()
        urls = [x.strip() for x in lines]
    return urls


def time_experiment():
     """Performs the time experiment with LOC1 dataset (Section 5B of the paper)"""
    dataset = 'loc1'
    data_dir = join(DATA_DIR, dataset)
    pickle_path = join(CLASSIF_DIR, '%s.pickle' % dataset)
    urls = get_url_list(ALL_URL_LIST)

    if os.path.isfile(pickle_path):
        df = load_data(path=pickle_path)
    else:
        df = load_data(path=data_dir)

    num_samples = 20
    num_classes = 1500
    
    #Set training and test intervals
    test_int = ("21-10-18", "03-11-18")
    test_interval = (pd.to_datetime(test_int[0], dayfirst=True), pd.to_datetime(test_int[1], dayfirst=True))
    test_df = get_interval(df, test_interval[0], test_interval[1])
    training_intervals = [("26-08-18", "08-09-18"), ("09-09-18", "22-09-18"), ("23-09-18", "06-10-18"), ("07-10-18", "20-10-18"), ("21-10-18", "03-11-18")]
    training_intervals = [(pd.to_datetime(x, dayfirst=True), pd.to_datetime(y, dayfirst=True)) for x, y in training_intervals]
    
    for i in range(0, len(training_intervals)):
        print "Testting:", test_interval
        training_df = get_interval(df, training_intervals[i][0], training_intervals[i][1])
        training_df_trimmed, test_df_trimmed = trim_cross_comparison(training_df, test_df, num_samples, num_classes)
        print "Start cross validation for:", training_intervals[i]
        start = time.time()
        tag = "_interval_test5_" + str(i)
        results, _ = cross_classify(training_df_trimmed, test_df_trimmed, OUTPUT_ACC + tag, OUTPUT_ACC + tag)
        report = describe_classif_reports(results, OUTPUT_TP + tag + "_")
        print_stats_paper(report, OUTPUT_STATISTICS + tag)
        with open(OUTPUT_REPORT + tag, "w") as f:
            f.write(report.to_string())
        stop = time.time()
        print "Time taken:", stop - start


def rpi_experiment(remove_bad=False):
    """Performs the Desktop vs RPi experiment with LOC1 and RPI datasets (Section 5E of the paper)"""

    #Input data for RPi
    dataset = 'rpi'
    pickd = 'rpi'
    data_dir_pi = join(DATA_DIR, dataset)
    pickle_path_pi = join(DATA_DIR, '%s.pickle' % pickd)
    #Input data for Desktop
    dataset = 'loc1'
    pickd = 'loc1'
    data_dir = join(DATA_DIR, dataset)
    pickle_path = join(DATA_DIR, '%s.pickle' % pickd)

    urls = get_url_list(ALL_URL_LIST)

    #Output files. D = Desktop, R = Rpi, Tag DR means data trained on Desktop and tested on RPi (and vice-versa)
    OUTPUT_STATS_DR = join(EXP_RESULTS_DIR, 'statsdr') #precision/recall/f-score stats
    OUTPUT_STATS_RD = join(EXP_RESULTS_DIR, 'statsrd') #precision/recall/f-score stats
    OUTPUT_REPORT_DR = join(EXP_RESULTS_DIR, 'reportdr') #detailed report
    OUTPUT_REPORT_RD = join(EXP_RESULTS_DIR, 'reportrd') #detailed report
    OUTPUT_TP_DR = join(EXP_RESULTS_DIR, 'tpdr_') #true vs predicted label (for other analysis if required)
    OUTPUT_TP_RD = join(EXP_RESULTS_DIR, 'tprd_') #true vs predicted label (for other analysis if required)
    OUTPUT_ACC_DR = join(EXP_RESULTS_DIR, 'accdr') #accuracy for each fold
    OUTPUT_ACC_RD = join(EXP_RESULTS_DIR, 'accrd') #accuracy for each fold
   
    num_samples = 60
    num_classes = 700

    if os.path.isfile(pickle_path):
        df = load_data(path=pickle_path)
    else:
        df = load_data(path=data_dir)
    if os.path.isfile(pickle_path_pi):
        df_pi = load_data(path=pickle_path_pi)
    else:
        df_pi = load_data(path=data_dir_pi)

    #to remove bad webpages
    if remove_bad:
        print "delete strange webpages"
        df = remove_strange(df, STRANGE_URL_LIST, urls)
        df_pi = remove_strange(df_pi, STRANGE_URL_LIST, urls)

    #Code to modify sizes of RPi packets. Uncomment if required.
    #delta = 8
    #df_pi["lengths"] = df_pi["lengths"] + delta

    print "Trim data to required number of samples and classes"
    desktop, rpi = trim_cross_comparison(df, df_pi, num_samples, num_classes)
    assert_dataset_size(desktop, num_samples, num_classes)
    assert_dataset_size(rpi, num_samples, num_classes)
   
    print "Start cross validation"
    start = time.time()
    results_desk, results_rpi = cross_classify(desktop, rpi, output_accd, output_accr)
    print "Write results"
    report_desk = describe_classif_reports(results_desk, output_tpd)
    report_rpi = describe_classif_reports(results_rpi, output_tpr)
    print_stats_paper(report_desk, output_statisticsd)
    print_stats_paper(report_rpi, output_statisticsr)
    with open(output_reportd, "w") as f:
        f.write(report_desk.to_string())
    with open(output_reportr, "w") as f:
        f.write(report_rpi.to_string())
    stop = time.time()
    print "Total time taken:", stop - start


def normal_experiment(remove_bad=False):
    """Performs the base experiment with LOC1 dataset (Section 5A of the paper)"""

    dataset = 'loc1'
    data_dir = join(DATA_DIR, dataset)
    pickle_path = join(DATA_DIR, '%s.pickle' % dataset)
    urls = get_url_list(ALL_URL_LIST)
    num_classes = 1500
    num_samples = 60

    if os.path.isfile(pickle_path):
        df = load_data(path=pickle_path)
    else:
        df = load_data(path=data_dir)

    #To remove bad webpages
    if remove_bad:
        print "Delete strange webpages"
        df = remove_strange(df, STRANGE_URL_LIST, urls)

    print "Trim data to required number of samples and classes"
    df_trimmed = trim_sample_df(df, num_samples, map(str, range(num_classes)))
    print "Start cross validation"
    start = time.time()
    results = cross_validate(df_trimmed, OUTPUT_ACC)
    print "Write results"
    report = describe_classif_reports(results, OUTPUT_TP)
    print_stats_paper(report, OUTPUT_STATS)
    with open(OUTPUT_REPORT, "w") as f:
        f.write(report.to_string())
    stop = time.time()
    print "Total time taken:", stop - start

if __name__ == '__main__':
    # parse args
    def help():
        print "python classify_pipeline.py [-h] EXP_NAME"
        print "\tEXP_NAME\tspecify an experiment: \'rpi\', \'normal\', \'time\'"
        print "\t-h\t\tshows this message"

    if '-h' in sys.argv:
        help()
        sys.exit(-1)

    if len(sys.argv) != 2:
        print "ERROR: Incorrect number of arguments"
        help()
        sys.exit(-1)

    exp_name = sys.argv[1]

    # instantiate paths specific to experiment
    global EXP_RESULTS_DIR, OUTPUT_STATS, OUTPUT_REPORT, OUTPUT_TP, OUTPUT_ACC
    EXP_RESULTS_DIR = join(RESULTS_DIR, '%s') % exp_name
    OUTPUT_STATS = join(EXP_RESULTS_DIR, 'stats')
    OUTPUT_REPORT = join(EXP_RESULTS_DIR, 'report')
    OUTPUT_TP = join(EXP_RESULTS_DIR, 'tp_')
    OUTPUT_ACC = join(EXP_RESULTS_DIR, 'acc')

    # make dir
    if not os.path.isdir(EXP_RESULTS_DIR):
        os.makedirs(EXP_RESULTS_DIR)

    # run experiment
    if exp_name == 'normal':
        normal_experiment()

    elif exp_name == 'rpi':
        rpi_experiment()

    elif exp_name == 'time':
        time_experiment()
