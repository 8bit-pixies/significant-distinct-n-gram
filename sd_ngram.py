"""
Significant and Distinctive n-Grams

Should probably use gensim for purposes of having a partial-fit 
variation so that it can be updated on a stream?
"""

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile


class SDNgram(TransformerMixin):
    def __init__(self, max_ngram=3, omega=10, tfidf_config={"min_df": 5}):
        self.ngram_process = range(1, max_ngram + 1)
        self.omega = omega
        self.tfidf_config = tfidf_config
        self.tfidf_vectorizers = []
        self.vocab = []

    def transform(self, X):
        return X

    def fit(self, X, y=None):
        for i in self.ngram_process:
            if i == 1:
                temp = TfidfVectorizer(ngram_range=(i, i), **self.tfidf_config)
                temp.fit(X)
                X_temp = temp.transform(X)

                sp = SelectPercentile(
                    score_func=lambda X, _: -np.sum(X, axis=0), percentile=self.omega
                )
                # print(X_temp.A)
                sp.fit(X_temp.A, X_temp.A)
                inv_map = {v: k for k, v in temp.vocabulary_.items()}
                self.vocab = [[inv_map[x] for x in np.nonzero(sp.get_support())[0]]]
                self.vocab_last = list(
                    set([inv_map[x] for x in np.nonzero(sp.get_support())[0]])
                )

                self.tfidf_vectorizers.append(
                    TfidfVectorizer(ngram_range=(i, i), vocabulary=self.vocab_last).fit(
                        X
                    )
                )
                del temp
            else:
                tfidf_temp = TfidfVectorizer(ngram_range=(i, i), **self.tfidf_config)
                tfidf_temp.fit(X)
                # fix the vocab if it exists above.

                vocab_current = list(tfidf_temp.vocabulary_.keys())
                vocab = []
                for curr_v in vocab_current:
                    for old_v in self.vocab_last:
                        if old_v in curr_v:
                            vocab.append(curr_v)
                            break
                vocab = list(set(vocab))
                temp = TfidfVectorizer(ngram_range=(i, i), vocabulary=vocab)
                temp.fit(X)
                X_temp = temp.transform(X)
                sp = SelectPercentile(
                    score_func=lambda X, _: -np.sum(X, axis=0), percentile=self.omega
                )
                sp.fit(X_temp.A, X_temp.A)
                inv_map = {v: k for k, v in temp.vocabulary_.items()}
                self.vocab.append([inv_map[x] for x in np.nonzero(sp.get_support())[0]])
                self.vocab_last = list(
                    set([inv_map[x] for x in np.nonzero(sp.get_support())[0]])
                )
                del tfidf_temp
                del temp

                if len(self.vocab_last) > 0:
                    self.tfidf_vectorizers.append(
                        TfidfVectorizer(
                            ngram_range=(i, i), vocabulary=self.vocab_last
                        ).fit(X)
                    )
                else:
                    break

        return self

    def transform(self, X):
        X_ = []
        for tfidf in self.tfidf_vectorizers:
            X_.append(tfidf.transform(X))
        return X_


if __name__ == "__main__":
    categories = ["alt.atheism", "talk.religion.misc", "comp.graphics", "sci.space"]
    newsgroups_train = fetch_20newsgroups(subset="train", categories=categories)

    sdn = SDNgram(max_ngram=5)
    nt_data = sdn.fit_transform(newsgroups_train.data)
    print(sdn.tfidf_vectorizers)
    # there shouldl be a final post processing to remove stop words...exercise for later
