"""balanced_accuracy_score"""
import typing
import numpy as np
from h2oaicore.metrics import CustomScorer
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder


class balancedaccuracy(CustomScorer):
    _description = "Balanced Accuracy Score"
    _unsupervised = False  # ignores actual, uses predicted and X to compute metrics
    _regression = False
    _binary = True
    _multiclass = False
    _maximize = True
    _perfect_score = 1.0
    _supports_sample_weight = True  
    _display_name = "Balanced Accuracy"
    
    @staticmethod
    def do_acceptance_test():
        """
        Whether to enable acceptance tests during upload of recipe and during start of Driverless AI.
        Acceptance tests perform a number of sanity checks on small data, and attempt to provide helpful instructions
        for how to fix any potential issues. Disable if your recipe requires specific data or won't work on random data.
        """
        return False

    def score(self,
              actual: np.array,
              predicted: np.array,
              sample_weight: typing.Optional[np.array] = None,
              labels: typing.Optional[np.array] = None,
              **kwargs) -> float:
        lb = LabelEncoder()
        labels = lb.fit_transform(labels)
        actual = lb.transform(actual)
        return balanced_accuracy_score(actual, predicted, sample_weight)
