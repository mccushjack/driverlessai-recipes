"""Target-encode numbers by their logarithm"""
import math

from h2oaicore.transformer_utils import CustomTransformer
import datatable as dt
import numpy as np
from h2oaicore.transformers import CVTargetEncodeTransformer
from sklearn.preprocessing import LabelEncoder


class LogScaleBinner:
    def fit_transform(self, X: dt.Frame):
        return self.transform(X)

    def transform(self, X: dt.Frame):
        assert X.ncols == 1
        return X[:, dt.stype.str32(dt.stype.int32(dt.log(dt.f[0])))]


class LogScaleTargetEncodingTransformer(CustomTransformer):
    _testing_can_skip_failure = False  # ensure tested as if shouldn't fail

    @staticmethod
    def get_default_properties():
        return dict(col_type="numeric", min_cols=1, max_cols=1, relative_importance=1)

    def fit_transform(self, X: dt.Frame, y: np.array = None):
        # Roughly: Convert numbers to a string of their exponent
        self.binner = LogScaleBinner()
        X = self.binner.fit_transform(X)

        # Compute mean target (out of fold) per same string
        self.cvte = CVTargetEncodeTransformer(cat_cols=X.names)

        if self.labels is not None:
            # for classification, always turn y into numeric form, even if already integer
            y = dt.Frame(LabelEncoder().fit(self.labels).transform(y))

        X = dt.Frame(self.cvte.fit_transform(X, y))
        # ensure no inf
        # Don't leave inf/-inf
        for i in range(X.ncols):
            X.replace([math.inf, -math.inf], None)
        return X

    def transform(self, X: dt.Frame):
        X = self.binner.transform(X)
        X = dt.Frame(self.cvte.transform(X))
        # Don't leave inf/-inf
        for i in range(X.ncols):
            X.replace([math.inf, -math.inf], None)
        return X
