import re
import os
from os.path import join, dirname, abspath, pardir, basename, normpath
import json

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display as disp

from multiprocessing import cpu_count
from joblib import Parallel, delayed
from functools import partial

BASE_DIR = abspath(join(dirname(__file__), pardir, pardir, pardir))
CODE_DIR = join(BASE_DIR, 'code')
COLLEC_DIR = join(CODE_DIR, 'collection')
ALL_URL_LIST = join(COLLEC_DIR, 'short_list_500')

# MATH FUNCTIONS:
def harmonic_mean(x, y, factor=1.0):
    """Returns the weighter harmonic mean of x and y.

    `factor` allows to express how many times we value `x`
    over `y` in the harmonic mean. This is equivalent to
    the F_{\beta} score:
        https://en.wikipedia.org/wiki/F1_score

    Important: we are assuming x > 0 and y > 0 here.
    """
    assert x > 0 and y > 0
    x, y = float(x), float(y)  # cast to floats
    factor2 = factor ** 2
    return (x * y * (1 + factor2)) / ((factor2 * x) + y)


def round_mult(x, base=10):
    """Round to nearest multiple of base."""
    a = ((abs(x) - 1) // base) * base  # nearest multiple
    b = a + base  # round up
    return b * np.sign(x)


# DATASET FORMATTING FUNCTIONS
def trim(elements, n):
    """Select `n` number of elements.

    If there are equal or greater than `n` elements,
    select `n` elements as `True`. Otherwise, leave them untouched,
    as we assume they are initialized to `False`.
    """
    if len(elements) >= n:    # if there are enough elements,
        elements[:n] = True   # set `n` to `True` and leave
    return elements           # the rest to `False`.


def trim_df(df, num_insts):
    """Return a dataframe with the same number of instances per class.

    The dataframe, `df`, has a field with the class id called `class_label`.
    """
    df2 = df.copy()  # the selected field should not appear in the original `df`
    df2['selected'] = False  # initialize all instances to not selected
    classes = df2.groupby('class_label')  # group instances by class
    trim_part = partial(trim, n=num_insts)  # partial trim to n=NUM_INSTS
    df2['selected'] = classes.selected.transform(trim_part)  # mark as selected
    selected = df[df2.selected]  # get the selected instances
    return selected


def sample_classes(df, classes=None):
    if type(classes) is int:
        sample = random.sample(df.class_label.unique(), classes)
    elif type(classes) is list:
        sample = classes
    else:
        raise Exception("Type of classes not recognized.")
    selected_classes = df.class_label.isin(sample)
    return df[selected_classes]


def trim_cross_comparison(df1, df2, num_insts, num_classes):
    # remove classes that have less than `num_insts`
    df1_trimmed = trim_df(df1, num_insts)
    df2_trimmed = trim_df(df2, num_insts)
    result_insts1 = get_num_instances(df1_trimmed).unique()[0]
    result_insts2 = get_num_instances(df2_trimmed).unique()[0]
    print "Num instances df1, df2:", result_insts1, result_insts2

    # take a sample from the classes left that are in common
    df1_cl = set(df1_trimmed.class_label)
    df2_cl = set(df2_trimmed.class_label)
    intersection_cl = df1_cl.intersection(df2_cl)
    classes = random.sample(intersection_cl, num_classes)
    print "Num classes in common:", len(intersection_cl)

    # sample the instances that belongs to the list of selected classes
    df1_sampled = sample_classes(df1_trimmed, classes)
    df2_sampled = sample_classes(df2_trimmed, classes)

    # sort and re-index
    df1_sampled = df1_sampled.sort_values('class_label')
    df1_sampled.index = range(len(df1_sampled.index))
    df2_sampled = df2_sampled.sort_values('class_label')
    df2_sampled.index = range(len(df2_sampled.index))

    result_classes1 = set(df1_sampled.class_label)
    result_classes2 = set(df2_sampled.class_label)
    diff_classes = set(result_classes1).difference(result_classes2)
    print "Difference in classes", len(diff_classes)
    assert len(diff_classes) == 0

    return df1_sampled, df2_sampled


def trim_sample_df(df, num_insts, classes):
    df = trim_df(df, num_insts)
    df = sample_classes(df, classes)

    # restart index
    df = df.sort_values('class_label')
    df.index = range(len(df.index))

    return df


def assert_dataset_size(df, num_insts, num_classes):
    df_num_insts = get_num_instances(df).unique()[0]
    df_num_classes = get_num_classes(df)
    print df_num_insts, "==", num_insts
    print df_num_classes, "==", num_classes
    assert df_num_insts == num_insts
    assert df_num_classes == num_classes


def get_num_instances(df):
    """Return number of instances per class in the dataframe."""
    non_nan = df.dropna(axis='columns')  # nan cols would not have valid counts
    classes = non_nan.groupby('class_label')
    counts = classes.count()  # count instances in each group (class)
    first_column = counts.iloc[:, 1]  # we could get any column instead
    return first_column


def get_num_classes(df):
    """Return number of classes in the dataframe."""
    classes = df.groupby('class_label')
    return classes.ngroups


def optimal_instances_per_class(df, factor=1.0, draw=False):
    """Return 'optimal' number of instances per class.

    Find number of instances per class that maximizes both number of instances
    and number of classes. We use the harmonic mean to penalize individual
    extreme values.

    For that we use the histogram for the number of instances to obtain the
    the number of classes that have x instances.
    """
    # `bincount` returns the number of instances we have for each website
    counts = np.bincount(df.class_label.tolist())
    hist, bin_edges = np.histogram(counts)
    if draw:
        inst_counts = get_num_instances(df)
        inst_counts.hist(cumulative=-1, bins=100)
        plt.xlabel('Num of instances')
        plt.ylabel('Num of classes with x or more insts')
        plt.show()

    # scale the y-axis
    dx = bin_edges[1] - bin_edges[0]
    cum_hist = np.cumsum(hist) * dx

    # get the inverse cumulative sum
    inv_cum_hist = max(cum_hist) - cum_hist

    # compute the harmonic mean of tuples (y=f(x), x)
    hms = [harmonic_mean(x, y, factor) if y > 0 and x > 0 else 0
           for x, y in zip(bin_edges[1:], inv_cum_hist)]

    print hms

    # find index for max harmonic mean
    i = np.argmax(hms)

    # this is the optimal number of instances:
    opt_num_insts = int(bin_edges[i])

    # which leaves us with this number of classes:
    opt_num_classes = len(counts[counts >= opt_num_insts])

    if draw:
        print "Optimal number of instances:", opt_num_insts
        print "Optimal number of classes:", opt_num_classes

    return opt_num_insts, opt_num_classes


def list_array(column, pad=None, pad_with=0):
    """Return an array from a dataframe column with lists of the same size."""
    if pad is not None:  # in that case it's the array of lists' lengths
        mask = np.arange(pad.max()) < pad[:, None]
        if pad_with == 0:
            arrays = np.zeros(mask.shape, dtype=column.dtype)
        else:
            arrays = np.empty(mask.shape, dtype=column.dtype)
            arrays[:] = pad_with
        arrays[mask] = np.concatenate(column.values)
    else:
        arrays = column.values.tolist()
    return np.array(arrays)


def concat_lists(column):
    """Return list from concatenating all lists in the column."""
    arrays = list_array(column)
    return np.concatenate(arrays)


# OUTLIER REMOVAL
def min_inst(df, n=1):
    """Return only classes with at least one instance."""
    classes = df.groupby('class_label')
    counts = classes.inst.transform('count')
    sel_classes = df[counts > n]
    return sel_classes


def inst_class_stats(df, col='num_pkts'):
    """Get statistics about number of instances per class."""
    classes = df.groupby('class_label')
    stat = classes[col].describe()
    return stat


def std_thres(df, th=5):
    """Discard classes that have greater than `th` std."""
    stat = inst_class_stats(df)  # num of inst/class stats
    thresholded = stat[stat['std'] < th]
    class_labels = thresholded.reset_index().class_label
    sel_classes = df[df.class_label.isin(class_labels)]
    return sel_classes


# FEATURE FUNCTIONS
def get_bursts_per_class(df):
    classes = df.groupby('class_label')
    bursts = classes.bursts.apply(concat_lists)
    return bursts


def get_lengths_per_class(df):
    classes = df.groupby('class_label')
    lengths = classes.lengths.apply(concat_lists)
    return lengths


def get_uniq_len_count(lengths, all_lengths):
    """Return histogram of lengths over all possible lengths."""
    all_lengths = np.sort(all_lengths)  # sort array of all possible lengths
    bins = np.append(all_lengths, all_lengths[-1] + 1)
    return np.histogram(lengths, bins)[0]


def recover_order(sent_lengths, received_lengths, order):
    """Return sequence of lengths from snd/rcv lengths and order.

    Example:
        sent = [20, 33, 40]
        received = [33, 20, 20]
        order = [1, -1, 1, 1, -1, -1]
        Returns: [20, -33, 33, 40, -20, -20]
    """
    sequence = np.zeros(len(order))
    sequence[np.argwhere(order > 0).flatten()] = sent_lengths
    sequence[np.argwhere(order < 0).flatten()] = np.negative(received_lengths)
    return sequence


def get_bursts(len_seq):
    """Returns the sequence split by bursts.

    Example:
        len_seq = [20, -33, 33, 40, -20, -20]
        Returns: [[20], [-33], [33, 40], [-20, -20]]
    """
    directions = len_seq / abs(len_seq)
    index_dir_change = np.where(directions[1:] - directions[:-1] != 0)[0] + 1
    bursts = np.split(len_seq, index_dir_change)
    return bursts


def ngrams_bursts(len_seq, round=None):
    """Return sequence of bursts from sequence of lengths.

    The sequence of bursts is represented as specified for Dyer's VNG.

    Example:
        len_seq = [20, -33, 33, 40, -20, -20]
        Returns: [20, -33, 73, -40]
    """
    bursts = get_bursts(len_seq)
    ngrams = np.array(map(sum, bursts))
    if round is not None:
        ngrams = round_mult(ngrams, base=round)
    return ngrams


def join_str(lengths):
    return ' '.join(map(str, lengths))


## tsfresh format functions
def stack_lists(df_col):
    stacked = df_col.apply(pd.Series).stack()
    dropped = stacked.reset_index(level=1, drop=True).reset_index()
    return dropped.rename(columns={'index': 'id', 0: 'value'})


def stack_lists_df(df, col):
    if col == 'sent':
        df_col = df.lengths.apply(lambda x: x[np.where(x > 0)])
    else:
        df_col = df.lengths.apply(lambda x: np.abs(x[np.where(x < 0)]))
    stacked = stack_lists(df_col)
    stacked['kind'] = col
    return stacked


def convert(df):
    sent = stack_lists_df(df, 'sent')
    received = stack_lists_df(df, 'received')
    return pd.concat([sent, received])


# PARSING OF FILES
# regular expression used to parse files with traffic traces
PATH_REGEX = {'name': r'(?P<name>\w+)',
              'dev': r'(?:(?P<dev>[^_]+)_)?',
              'sites': r'(?:(?P<sites>[^_]+)_)?',
              'date': r'(?P<date>\d\d-\d\d-\d\d)',
              'inst': r'(?:_(?P<inst>\d+))?'}
#TRACE_PATH = os.path.join('traces', '{vm}{dev}{sites}{date}{inst}')
FNAME_REGEX = re.compile('{name}/{dev}{sites}{date}{inst}'.format(**PATH_REGEX))

# paths
BASE_DIR = abspath(join(dirname(__file__), pardir, pardir, pardir))
DATA_DIR = join(BASE_DIR, 'dataset')
PICKLE_DIR = join(DATA_DIR, 'pickles')
DEFAULT_PICKLE_FILE = join(PICKLE_DIR, 'index.pickle')

def load_data(path=DEFAULT_PICKLE_FILE, pickle=True):
    """Load dataset.

    If `path` is a file that exists, it should be a pickle and we load it.
    Otherwise, it should be a directory and we parse it.
    """
    if type(path) is list:
        dfs = [load_data(p, pickle='%s.pickle' % os.path.basename(p))
               for p in path]
        return pd.concat(dfs)
    elif type(path) is str:
        print "Loading", path
        if os.path.isfile(path):
            df = pd.read_pickle(path)
        else:
            df = parse_directory(path)
            dataset = basename(normpath(path))
            PICKLE_FILE = join(PICKLE_DIR, '%s.pickle' % dataset)
            if pickle:
                pickle_path = PICKLE_FILE
                if type(pickle) is str:
                    pickle_path = pickle
                print "Pickling to", pickle_path
                df.to_pickle(pickle_path)
        return df


def it_webpages(fpath):
    """Iterate over all the websites contained in a file."""
    with open(fpath) as f:
        data_dict = json.loads(f.read())
        try:
            for pcap_filename, values in data_dict.iteritems():
                webpage_num = pcap_filename[:-5]
                snd, rcv = values['sent'], values['received']
                order = values['order']
                lengths = recover_order(*map(np.array, [snd, rcv, order]))
                yield webpage_num, lengths
        except KeyError:
            print fpath, "does not have a known order sequence"
            return
            yield
        except Exception as e:
            print "ERROR:", fpath, pcap_filename, e


def sel_files(dpath):
    """Yield files that satisfy conditions."""
    sel_files = []
    for root, _, files in os.walk(dpath):
        for fname in files:
            if not fname.endswith('.json'):  # skip non-json files
                continue
            fpath = os.path.join(root, fname)
            sel_files.append(fpath)
    return sel_files

def parse_directory(dpath):
    """Traverse the directory and parse all the captures in it.

    Returns a dataframe containing encoded lengths.
    """
    print "Starting to parse"
    selected_files = sel_files(dpath)
    print "Number of selected files", len(selected_files)

    # iterave over selected files and build dataframe
    empties = 0
    idx = pd.DataFrame(columns=PATH_REGEX.keys())
    for fpath in selected_files:
        m = FNAME_REGEX.search(fpath)
        if m is None:
            print "ERROR:", fpath, FNAME_REGEX.pattern
            continue
        row_head = {k: m.group(k) for k in PATH_REGEX.iterkeys()}
        for i, (webpage_id, lengths) in enumerate(it_webpages(fpath)):
            if len(lengths) == 0:
                empties += 1
                continue
            row_head['fname'] = os.path.basename(fpath)
            row_head['class_label'] = webpage_id
            row_head['lengths'] = lengths
            idx = idx.append(row_head, ignore_index=True)
        print i, 'sites in', fpath
    print "Empty traces:", empties

    # fix some naming issues:
    idx['inst'] = idx.inst.fillna(0)
    idx['date'] = pd.to_datetime(idx.date.str.replace('-18', '-2018'),
                                 format='%d-%m-%Y')
    #idx['dev'] = idx.dev.replace('browse', 'desktop')
    #idx.loc[idx.sites == 'desktop', ['dev', 'sites']] = ['desktop', None]
    return idx


# OPTIMIZATION
def apply_parallel(dfGrouped, func):
    retLst = Parallel(n_jobs=cpu_count())(delayed(func)(group) for name, group in dfGrouped)
    return pd.concat(retLst)


# OTHER UTILS
def load_mapping():
    """Return Alexa as a list."""
    return [l.strip() for l in open(ALL_URL_LIST)]
ALEXA_MAP = load_mapping()


def alexa_rank(url):
    """Return index in Alexa."""
    return ALEXA_MAP.index(url)


def url(index):
    """Return URL for the index in Alexa."""
    return ALEXA_MAP[index]


def display(df, urls=False):
    """Redefine display to show URLs instead of indices."""
    dft = df
    if urls:
        dft = df.copy()
        if df.index.name == 'class_label':
            dft = dft.reset_index()
            if 'index' in dft.columns:
                dft = dft.drop(['index'], axis=1)
        if 'class_label' in dft.columns:
            dft['class_label'] = dft.class_label.apply(lambda x: url(int(x)))
        if df.index.name == 'class_label':
            dft.set_index('class_label')
    disp(dft)


# identify function handle
def ident(x):
    return x
