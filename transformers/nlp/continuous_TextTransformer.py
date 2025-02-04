"""Creates a TF-IDF based text transformation that can be continuously updated with new data and vocabulary."""

import importlib
from h2oaicore.transformer_utils import CustomTransformer
from h2oaicore.transformers import TextTransformer, CPUTruncatedSVD
import datatable as dt
import numpy as np
from h2oaicore.systemutils import config, remove, user_dir
import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
import ast
import copy
import scipy as sc
import pandas as pd


def get_value(config, key):
    if key in config.recipe_dict:
        return config.recipe_dict[key]
    elif "config_overrides" in config.get_overrides_dict():
        data = config.get_overrides_dict()["config_overrides"]
        data = ast.literal_eval(ast.literal_eval(data))
        return data.get(key, None)
    else:
        return None


# """
# {
# 'Custom_TextTransformer_load':/home/dmitry/Desktop/tmp/save_000.pkl',
# 'Custom_TextTransformer_save':'/home/dmitry/Desktop/tmp/save_001.pkl'
# }
# """

# "{'Custom_TextTransformer_load':'/home/dmitry/Desktop/tmp/save_000.pkl','Custom_TextTransformer_save':'/home/dmitry/Desktop/tmp/save_001.pkl'}"


class Cached_TextTransformer(CustomTransformer):
    _regression = True
    _binary = True
    _multiclass = True
    _testing_can_skip_failure = False  # ensure tested as if shouldn't fail

    _display_name = "Cached_TextTransformer"
    load_key = "Custom_TextTransformer_load"
    save_key = "Custom_TextTransformer_save"

    _can_use_gpu = False
    _can_use_multi_gpu = False

    @staticmethod
    def do_acceptance_test():
        return False

    @staticmethod
    def get_parameter_choices():
        return {
            "max_features": [None],
            "tf_idf": [True, False],
            "max_ngram": [1, 2, 3],
            "dim_reduction": [50],
        }

    @staticmethod
    def get_default_properties():
        return dict(col_type="text", min_cols=1, max_cols=1, relative_importance=1)

    def __init__(
        self, max_features=None, tf_idf=True, max_ngram=1, dim_reduction=50, **kwargs
    ):
        super().__init__(**kwargs)

        self.loaded = False

        self.load_path = get_value(config, self.load_key)
        self.save_path = get_value(config, self.save_key)

        if not self.load_path:
            self.TextTransformer = TextTransformer(
                max_features=max_features,
                tf_idf=tf_idf,
                max_ngram=max_ngram,
                dim_reduction=dim_reduction,
                **kwargs
            )
            self.TextTransformer._can_use_gpu = self._can_use_gpu
            self.TextTransformer._can_use_multi_gpu = self._can_use_multi_gpu
        else:
            data = joblib.load(self.load_path)
            if isinstance(data, dict):
                self.TextTransformer = data["txtTransformer"]
                self.tf_idf = data["tf_idf"]
                self.target = data["target"]
            else:
                self.TextTransformer = data
                self.tf_idf = {}
                self.target = None
            self.loaded = True
            self.TextTransformer._can_use_gpu = self._can_use_gpu
            self.TextTransformer._can_use_multi_gpu = self._can_use_multi_gpu

    def fit_transform(self, X: dt.Frame, y: np.array = None):
        self.TextTransformer.N_ = X.shape[0]
        result = self.TextTransformer.fit_transform(X.to_pandas())

        if self.save_path:
            joblib.dump(self.TextTransformer, self.save_path)
        return result

    def transform(self, X: dt.Frame):
        return self.TextTransformer.transform(X.to_pandas())

    _mojo = True
    from h2oaicore.mojo import MojoWriter, MojoFrame

    def to_mojo(
        self, mojo: MojoWriter, iframe: MojoFrame, group_uuid=None, group_name=None
    ):
        return self.TextTransformer.write_to_mojo(mojo, iframe, group_uuid, group_name)


# class Updatable_TextTransformer_TFIDFOnly(Cached_TextTransformer):
#     """
#     Only updates TF-IDF terms, vocabulary and stop word list remain the same
#     """
#     _display_name = "Updatable_TextTransformer_TFIDFOnly"

#     @staticmethod
#     def inverse_idf(idf_, N_):
#         tmp = np.exp(idf_ - 1)
#         tmp = np.round((N_+1) / tmp) - 1
#         return tmp

#     def fit_transform(self, X: dt.Frame, y: np.array = None):
#         if self.loaded:
#             X_ = X.to_pandas()
#             N_ = len(X_)
#             for col in self.input_feature_names:
#                 if self.TextTransformer.tf_idf: # update tf-idf terms for tokens in new data
#                     cv = TfidfVectorizer()
#                     pre_trained = self.TextTransformer.pipes[col][0]["model"]
#                     cv.set_params(**pre_trained.get_params())
#                     cv.set_params(**{
#                         "vocabulary": pre_trained.vocabulary_,
#                         "stop_words": pre_trained.stop_words_
#                     })
#                     pipe_ = copy.deepcopy(self.TextTransformer.pipes[col][0])
#                     new_pipe = []
#                     for step in pipe_.steps:
#                         if step[0] != 'model':
#                             new_pipe.append(step)
#                         else:
#                             new_pipe.append(('model', cv))
#                             break
#                     new_pipe = Pipeline(new_pipe)
#                     new_pipe.fit(self.TextTransformer.stringify_col(X_[col]))

#                     freq2 = self.inverse_idf(cv.idf_, N_)

#                     freq = self.inverse_idf(
#                         pre_trained.idf_,
#                         self.TextTransformer.N_
#                     )
#                     freq = freq + freq2
#                     self.TextTransformer.N_ = self.TextTransformer.N_ + N_
#                     freq = np.log((self.TextTransformer.N_+1) / (1+freq)) + 1
#                     pre_trained.idf_ = freq

#             result = self.TextTransformer.transform(X.to_pandas())

#         else:
#             self.TextTransformer.N_ = X.shape[0]
#             result = self.TextTransformer.fit_transform(X.to_pandas())

#         if self.save_path:
#             joblib.dump(self.TextTransformer, self.save_path)
#         return result


class Updatable_TextTransformer(Cached_TextTransformer):
    """
    Updates TF-IDF terms, vocabulary and stop word, same for CountVectorizer
    Updates SVD matrix in order to incorporate new terms and adjust influence of old ones
    """

    _display_name = "Updatable_TextTransformer"

    @staticmethod
    def get_parameter_choices():
        dict_ = Cached_TextTransformer.get_parameter_choices()
        dict_["step"] = [1e-5, 1e-4, 1e-3, 1e-2, 0.1]
        return dict_

    def __init__(
        self,
        max_features=None,
        tf_idf=True,
        max_ngram=1,
        dim_reduction=50,
        step=0.1,
        **kwargs
    ):
        super().__init__(
            max_features=None, tf_idf=True, max_ngram=1, dim_reduction=50, **kwargs
        )

        self.step = step

    @staticmethod
    def inverse_idf(idf_, N_):
        tmp = np.exp(idf_ - 1)
        tmp = np.round((N_ + 1) / tmp) - 1
        return tmp

    def fit_transform(self, X: dt.Frame, y: np.array = None, append=False):
        y_ = y
        new_data = []
        if self.loaded:
            X_ = X.to_pandas()
            N_ = len(X_)
            for col in self.input_feature_names:
                if self.TextTransformer.tf_idf:
                    # train new TfidfVectorizer in order to expand vocabulary of the old one and adjust idf terms
                    cv = TfidfVectorizer()
                    pre_trained = self.TextTransformer.pipes[col][0]["model"]
                    cv.set_params(**pre_trained.get_params())
                    pipe_ = copy.deepcopy(self.TextTransformer.pipes[col][0])
                    new_pipe = []
                    for step in pipe_.steps:
                        if step[0] != "model":
                            new_pipe.append(step)
                        else:
                            new_pipe.append(("model", cv))
                            break
                    new_pipe = Pipeline(new_pipe)
                    new_pipe.fit(self.TextTransformer.stringify_col(X_[col]))

                    freq2 = self.inverse_idf(cv.idf_, N_)

                    freq = self.inverse_idf(pre_trained.idf_, self.TextTransformer.N_)

                    # adjust vocabulary and stop word list based on newly data
                    # adjust frequency terms and idf terms
                    new_freq = []
                    remapped_freq = np.zeros(len(freq))
                    dict_ = copy.copy(pre_trained.vocabulary_)
                    stop_list = copy.copy(pre_trained.stop_words_)
                    max_val = len(dict_)

                    for k in cv.vocabulary_:
                        val = dict_.get(k, -1)
                        if val == -1:
                            dict_[k] = max_val
                            existed = stop_list.discard(k)
                            max_val += 1
                            new_freq.append(freq2[cv.vocabulary_[k]])
                        else:
                            remapped_freq[val] = freq2[cv.vocabulary_[k]]

                    pre_trained.vocabulary_ = dict_
                    pre_trained.stop_words_ = stop_list

                    freq = freq + remapped_freq
                    freq = np.hstack([freq, new_freq])

                    self.TextTransformer.N_ = self.TextTransformer.N_ + N_
                    freq = np.log((self.TextTransformer.N_ + 1) / (1 + freq)) + 1
                    pre_trained.idf_ = freq

                else:
                    # train new CountVectorizer in order to expand vocabulary of the old one
                    cv = CountVectorizer()
                    pre_trained = self.TextTransformer.pipes[col][0]["model"]
                    cv.set_params(**pre_trained.get_params())
                    pipe_ = copy.deepcopy(self.TextTransformer.pipes[col][0])
                    new_pipe = []
                    for step in pipe_.steps:
                        if step[0] != "model":
                            new_pipe.append(step)
                        else:
                            new_pipe.append(("model", cv))
                            break
                    new_pipe = Pipeline(new_pipe)
                    new_pipe.fit(self.TextTransformer.stringify_col(X_[col]))

                    # adjust vocabulary and stop word list based on newly data
                    dict_ = copy.copy(pre_trained.vocabulary_)
                    stop_list = copy.copy(pre_trained.stop_words_)
                    max_val = len(dict_)
                    for k in cv.vocabulary_:
                        val = dict_.get(k, -1)
                        if val == -1:
                            dict_[k] = max_val
                            existed = stop_list.discard(k)
                            max_val += 1

                    pre_trained.vocabulary_ = dict_
                    pre_trained.stop_words_ = stop_list

                # get transformed data in order to adjust SVD matrix
                svd_ = self.TextTransformer.pipes[col][1]
                if isinstance(svd_, CPUTruncatedSVD):
                    X_transformed = self.TextTransformer.pipes[col][0].transform(
                        self.TextTransformer.stringify_col(X_[col])
                    )
                    if col in self.tf_idf:
                        # combine saved matrix with the new one
                        newCols = X_transformed.shape[1] - self.tf_idf[col].shape[1]
                        if newCols > 0:
                            newCols = np.zeros((self.tf_idf[col].shape[0], newCols))
                            new_tf_idf = sc.sparse.hstack([self.tf_idf[col], newCols])
                        else:
                            new_tf_idf = self.tf_idf[col]
                        new_tf_idf = sc.sparse.vstack([new_tf_idf, X_transformed])
                        self.tf_idf[col] = new_tf_idf
                        # fit SVD on combined matrix
                        new_svd = CPUTruncatedSVD()
                        new_svd.set_params(**svd_.get_params())
                        new_svd.fit(self.tf_idf[col])

                        # replace old svd matrix with new one
                        svd_.components_ = new_svd.components_

                        if append:
                            data_ = svd_.transform(self.tf_idf[col])
                            data_ = self.TextTransformer.pipes[col][2].transform(data_)
                            data_ = pd.DataFrame(
                                data_,
                                columns=self.TextTransformer.get_names(
                                    col, data_.shape[1]
                                ),
                            )
                            new_data.append(data_)

                    else:
                        self.tf_idf[col] = X_transformed
                        # train new SVD to get new transform matrix
                        new_svd = CPUTruncatedSVD()
                        new_svd.set_params(**svd_.get_params())
                        new_svd.fit(X_transformed)

                        # adjust old transform matrix based on new one
                        grad = (
                            svd_.components_
                            - new_svd.components_[:, : svd_.components_.shape[1]]
                        )
                        grad = self.step * grad
                        svd_.components_ = svd_.components_ - grad
                        svd_.components_ = np.hstack(
                            [
                                svd_.components_,
                                new_svd.components_[:, svd_.components_.shape[1] :],
                            ]
                        )

            if append:
                new_data = pd.concat(new_data, axis=1)
                if self.target is not None:
                    y_ = np.hstack([self.target, y_])

                if self.save_path:
                    joblib.dump(
                        {
                            "txtTransformer": self.TextTransformer,
                            "tf_idf": self.tf_idf,
                            "target": y_,
                        },
                        self.save_path,
                    )
                return new_data, y_

            result = self.TextTransformer.transform(X.to_pandas())

        else:

            self.TextTransformer.N_ = X.shape[0]
            result = self.TextTransformer.fit_transform(X.to_pandas())
            X_ = X.to_pandas()
            self.tf_idf = {}
            for col in self.input_feature_names:
                self.tf_idf[col] = self.TextTransformer.pipes[col][0].transform(
                    self.TextTransformer.stringify_col(X_[col])
                )

        if self.save_path:
            joblib.dump(
                {
                    "txtTransformer": self.TextTransformer,
                    "tf_idf": self.tf_idf,
                    "target": y_,
                },
                self.save_path,
            )
        return result
