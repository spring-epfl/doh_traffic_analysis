'''
Methods and classes to calculate entropies.
'''
from scipy import stats
import numpy as np
from util import *


def frequency(length_group):
    return length_group / float(length_group.sum())


def weight(class_label, p=0.5):
    r = float(class_label)
    return p ** (r + 1)


def weighted_sizes(group):
    label = group.class_label.iloc[0]
    p_w = weight(label)
    m_s = len(group)
    return m_s * p_w


def entropy(x):
    return -np.nansum(x * np.log2(x))


"""
Methods to calculate the conditional entropy H(W | S).

Start lookink at the method `cond_entropy`, that's the main entry point.

IMPORTANT: for all these methods we assume:

  i) the dataset contains the same number of instances for each class
  ii) the priors are uniform.

If these assumptions are met, computing the entropy can be done by counting
frequencies in the anonymity sets.
"""
def slice_as_tuples(df, position=None):
    """Get sequences up to a certain packet number."""
    tmp_df = df.copy()
    def _slice_tuples(lengths_list):
        list_slice = lengths_list[:position]
        lengths_tuple = tuple(list_slice)
        return lengths_tuple
    tmp_df['lengths'] = tmp_df.lengths.apply(_slice_tuples)
    return tmp_df


def sliding_window(df, max_position=-1):
    return [con_entropies(slice_as_tuples(df, p))
            for p in xrange(max_position)]


class ProbabilityModel(object):
    def __init__(self, df, l=None):
        self.df = slice_as_tuples(df, l)
        self.anon_sets = self.anonymity_sets(self.df)
        self.posterior = self.estimate_posterior()

    @staticmethod
    def anonymity_sets(df):
        length_groups = df.groupby('lengths')
        return length_groups.class_label.apply(set)

    def rank(self):
        anon_sets = self.anon_sets.reset_index()
        anon_sets['class_label'] = anon_sets.class_label.apply(tuple)
        return AnonimitySetCounts(anon_sets)

    def estimate_posterior(self, priors='uniform'):
        """Return dataframe of probabilities as estimated by frequencies."""
        # group rows that have same length and class label
        grouped = self.df.groupby(['lengths', 'class_label'], as_index=False)

        # compute frequencies  TODO: parallelize these applies
        if priors == 'uniform':
            # compute sizes of those groups
            group_sizes = grouped.size()

        elif priors == 'geometric':
            # compute sizes of those groups
            group_sizes = grouped.apply(weighted_sizes)

        # group sizes only by having the same length
        # TODO: parallelize this one?
        posterior = group_sizes.groupby(level=0).apply(frequency)
        return posterior

    def entropy(self):
        # compute entropies from probabilities for specific sequences
        # that is H(W | S=s1), ..., H(W | S=s_n)
        return self.posterior.groupby(level=0).apply(entropy)

    def cond_entropy(self):
        """Compute conditional entropy for whole dataset."""
        # get the probabilities of length sequences
        prob_length = self.df.groupby('lengths').size() / len(self.df)

        # compute conditional entropy
        return sum(prob_length * self.entropy())


class AnonimitySetCounts():
    def __init__(self, anon_sets):
        self.counts = self.get(anon_sets)

    @staticmethod
    def get(anon_sets):
        counts = anon_sets.groupby('class_label').count()
        counts['list_lengths'] = anon_sets.groupby('class_label').lengths.apply(list)
        counts = counts.reset_index()
        counts['num_sites'] = counts.class_label.apply(len)
        return counts

    def filter(self, min_counts=0):
        self.counts = self.counts[self.counts.num_sites > min_counts]
        return self

    def sort(self):
        self.counts = self.counts.sort_values('lengths', ascending=False)
        return self

    def __str__(self):
        to_display = self.counts.copy()
        to_display['class_label'] = to_display.class_label.apply(lambda x: map(url, map(int, x)))
        to_display['site_name_length'] = to_display.class_label.apply(lambda x: map(len, x))
        with pd.option_context('max_colwidth', -1):
            display(to_display)
        return 'Displayed'


# Unit tests for the functions above:

def test_is_P_is_prob_distrib():
    """Values should add up to 1."""
    P = get_prob_matrix(W, uS)

    # The probabilities in each row in P should sum one (approx)
    # the probabilities of a row of P are: P(w) * P(size_j | w)
    # for the website w in that row and all j in the range of sizes.
    assert P.shape == (nclasses, len(uS))
    ones = np.ones(P.shape[0]).astype(float)
    second_dec = ones / 100
    assert (np.abs(np.sum(P, axis=1) - ones) < second_dec).all()


def test_is_B_is_prob_distrib():
    """Values should add up to 1."""
    P = get_prob_matrix(W, uS)

    B = np.apply_along_axis(lambda x: bayes(x, nclasses), 0, P)

    # The probabilities in each column in B should sum one (approx)
    # the probabilities of a column in P are: P(w_i | size)
    # for the size in that column and all i in the range of websites.
    assert B.shape == (nclasses, len(uS))
    ones = np.ones(P.shape[1]).astype(float)
    second_dec = ones / 100
    assert (np.abs(np.sum(B, axis=0) - ones) < second_dec).all()


def test_bayes_method():
    p_col = np.array([4, 6, 10])

    # Bayes on this vector should do:
    # numerator = 1 / 2 * [4, 6, 10] = [2, 3, 5]
    # denominator =  sum([2, 3, 5]) = 10
    # B = [2, 3, 5] / 10 = [0.2, 0.3, 0.5]
    B = bayes(p_col, nclasses)
    ones = np.ones(p_col.shape[0]).astype(float)
    second_dec = ones / 100
    assert (np.abs(B - np.array([0.2, 0.3, 0.5])) < second_dec).all()

    # --- test matrix
    P = np.array([[0.3, 0.2, 0.5], [0.7, 0.1, 0.2]])
    B = np.apply_along_axis(lambda x: bayes(x, nclasses), 0, P)

    ones = np.ones(nclasses).astype(float)
    second_dec = ones / 100

    # Bayes on this matrix should do for first vector:
    # numerator = 1 / 2 * [0.3, 0.7] = [0.15, 0.35]
    # denominator =  sum([0.15, 0.35]) = 0.5
    # B = [0.15, 0.35] / 0.5 = [0.3, 0.7]
    assert (np.abs(B[:, 0] - np.array([0.3, 0.7])) < second_dec).all()

    # Bayes on this matrix should do for second vector:
    # numerator = 1 / 2 * [0.2, 0.1] = [0.1, 0.05]
    # denominator =  sum([0.1, 0.05]) = 0.15
    # B = [0.1, 0.05] / 0.15 = [0.66667, 0.33333]
    assert (np.abs(B[:, 1] - np.array([0.66667, 0.33333])) < second_dec).all()

    # Bayes on this matrix should do for second vector:
    # numerator = 1 / 2 * [0.5, 0.2] = [0.25, 0.1]
    # denominator =  sum([0.25, 0.1]) = 0.35
    # B = [0.25, 0.1] / 0.35 = [0.7142857, 0.285714]
    assert (np.abs(B[:, 2] - np.array([0.7142857, 0.285714])) < second_dec).all()


def test_all_traces_the_same():
    trace = [50, 100]
    test_df = pd.DataFrame({'class_label': map(str, range(nclasses)),
                            'lengths': [trace] * nclasses})
    test_df['num_pkts'] = test_df.lengths.apply(len)
    W, uS, cS = process_sizes(position, test_df, 2, 1)
    assert (W == np.array([[trace], [trace]])).all()

    v = get_size_count_vector(0, W, uS)
    # only one trace per site, counts should be one:
    assert (v == np.array([1])).all()

    P = get_prob_matrix(W, uS)
    # ...this, probability of that size give that website
    # should also be one
    assert (P == np.array([[1], [1]])).all()

    B = np.apply_along_axis(lambda x: bayes(x, nclasses), 0, P)
    # Now, probability of that website given that size
    # should be 0.5.
    assert (B == np.array([[0.5], [0.5]])).all()

    # Conditional entropy for 2 sites with identical traces
    # should be 1.0.
    cH = cond_entropy_from_bayes(B, cS)
    assert (cH == 1.0)


def test_all_traces_different():
    test_df = pd.DataFrame({'class_label': map(str, range(nclasses)),
                            'lengths': [[50, 100], [50, -50]]})
    test_df['num_pkts'] = test_df.lengths.apply(len)
    W, uS, cS = process_sizes(position, test_df, 2, 1)
    assert (W == np.array([[[50, 100]], [[50, -50]]])).all()

    v = get_size_count_vector(0, W, uS)
    # one size should have zero counts, and the other 1 count
    assert (v == np.array([0, 1])).all()

    P = get_prob_matrix(W, uS)
    # ...each website should have probability one for one size
    # and zero for the other.
    assert (P == np.array([[0, 1], [1, 0]])).all()

    B = np.apply_along_axis(lambda x: bayes(x, nclasses), 0, P)
    # Now, probability of that website given that size
    # should be 1.
    assert (B == np.array([[0, 1.], [1., 0]])).all()

    # Conditional entropy for 2 sites with all traces
    # different should be 0 (no uncertainty).
    cH = cond_entropy_from_bayes(B, cS)
    assert (cH == 0.)


def test_traces_different_length():
    test_df = pd.DataFrame({'class_label': map(str, range(nclasses)),
                            'lengths': [[50], [50, -50]]})
    test_df['num_pkts'] = test_df.lengths.apply(len)
    W, uS, cS = process_sizes(position, test_df, 2, 1)
    assert (W == np.array([[[50, np.inf]], [[50, -50]]])).all()

    v = get_size_count_vector(0, W, uS)
    # one size should have zero counts, and the other 1 count
    assert (v == np.array([0, 1])).all()

    P = get_prob_matrix(W, uS)
    # ...each website should have probability one for one size
    # and zero for the other.
    assert (P == np.array([[0, 1], [1, 0]])).all()

    B = np.apply_along_axis(lambda x: bayes(x, nclasses), 0, P)
    # Now, probability of that website given that size
    # should be 1.
    assert (B == np.array([[0, 1.], [1., 0]])).all()

    # Conditional entropy for 2 sites with all traces
    # different should be 0 (no uncertainty).
    cH = cond_entropy_from_bayes(B, cS)
    assert (cH == 0.)


def test_two_real_sites():
    # Facebook (2) and New York Times (126)
    test_df = df[df.class_label.isin(['2', '126'])].copy()

    # take only the 5 first lengths
    position = 5
    test_df['lengths'] = test_df.lengths.apply(lambda x: x[:position])
    test_df['num_pkts'] = test_df.lengths.apply(len)

    # verify that the intersection is empty
    fb = set(test_df[test_df.class_label == '2'].lengths.apply(tuple).tolist())
    ny = set(test_df[test_df.class_label == '126'].lengths.apply(tuple).tolist())
    assert len(fb.intersection(ny)) == 0

    # since intersection is zero, entropy should be zero
    H = cond_entropy(test_df)
    assert H == 0


def run_tests():
    test_bayes_method()
    test_is_P_is_prob_distrib()
    test_is_B_is_prob_distrib()
    test_all_traces_the_same()
    test_all_traces_different()
    test_traces_different_length()
    test_two_real_sites()


if __name__ == "__main__":
    run_tests()
