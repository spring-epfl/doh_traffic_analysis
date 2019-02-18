from tsfresh.feature_extraction import extract_features, MinimalFCParameters
from tsfresh.utilities.dataframe_functions import impute

from utils.util import *


class TSFreshBasicExtractor:
    def __init__(self):
        #print "Feature extraction: tsfresh basic"
        self.extracted_features = []

    def fit(self, x, y=None):
        return self

    def transform(self, df):
        df_stacked = convert(df)
        extracted_features = extract_features(df_stacked,
                                              column_id="id",
                                              column_kind="kind",
                                              column_value="value",
                                              default_fc_parameters=MinimalFCParameters())
        self.extracted_features = impute(extracted_features)
        return self.extracted_features.values.tolist()
