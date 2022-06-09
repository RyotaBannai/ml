import math
import os
import random
from collections import Counter

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA, PCA

from config import conf
from lib.utility import utils
from lib.tokenizer import tokenize


class Metadata:
    '''
    ニュースのCSVを読み込んで、学習用データの前処理を施していく。
    ニュース原稿を形態素解析にかけ、Counterを用いて出現頻度を数えておく。
    また各単語のIDFも前処理で計算して連想配列に保存しておく。
    '''

    def __init__(self):
        self.loaded_csv_paths = []
        self.token_to_id = {}
        # self.id_to_token = {}
        self.categories = set()
        self.category_dic = {}
        self.token_seq_no = 0
        self.doc_seq_no = 0
        self.token_to_docid = {}
        self.tokenized_news = []
        self.idf = {}

    def build(self, min_token_len: int):
        wakati_gen = tokenize.read_tokens()
        for category, tokens in wakati_gen:

            if min_token_len >= len(tokens):  #
                continue
            # print(category, tokens)
            token_counter = Counter(tokens)  #
            self.categories.add(category)  #
            self.update_token_dics(token_counter)  #
            self.tokenized_news.append([category, token_counter])  #
        self.update_idf()  #
        self.update_category_dic()

    def update_category_dic(self):
        """
        教師データを作成：one-hot
        カテゴリ数= 次元数（行列数）= 行列の長さ
        """
        cat_list = list(self.categories)
        cat_len = len(self.categories)
        for cat in cat_list:
            category_vec = [0] * cat_len
            category_vec[cat_list.index(cat)] = 1
            self.category_dic[cat] = category_vec

    def update_token_dics(self, token_counter: Counter):
        """
        tokenが現れるファイル番号を、
        tokenをkey、ファイル番号をvalue として作成
        """
        for token, _ in token_counter.items():
            if token not in self.token_to_id:
                self.token_to_id[token] = self.token_seq_no
                # self.id_to_token[self.token_seq_no] = token
                self.token_seq_no += 1
            self.token_to_docid.setdefault(token, set()).add(self.doc_seq_no)
        self.doc_seq_no += 1

    def update_idf(self) -> dict:
        self.idf = {token: self.__idf(docids)
                    for token, docids in self.token_to_docid.items()}

    def __idf(self, docids) -> float:
        return math.log(self.doc_seq_no / len(docids)) + 1


def tfidf(meta: Metadata, token_counter: Counter):
    '''
    TF-IDFベクトルを作成
    '''

    max_dim = meta.token_seq_no
    tf_vec = np.zeros(max_dim)
    total_number_of_token = sum(token_counter.values())
    for token, counter in token_counter.items():
        token_id = meta.token_to_id[token]
        tf_vec[token_id] = counter / total_number_of_token * meta.idf[token]

    return tf_vec


class PcaTfidfVectorizer:
    def __init__(self, meta: Metadata):
        self.meta = meta

    def incremental_fit(self, tokenized_news):
        # ipca = IncrementalPCA(n_components=conf.PCA_DIMENSION)
        pca = PCA(n_components=conf.PCA_DIMENSION)

        '''
          ニュース原稿のtfidfの主成分をbatch_sizeで指定した分ずつ求めていく
        '''
        # news_len = len(tokenized_news)
        random.shuffle(tokenized_news)
        # batch = conf.PCA_BATCH_DATA_LENGTH
        # for i in range(0, news_len, batch):
        #     chunks = tokenized_news[i:i + batch]
        #     mat = np.array([tfidf(self.meta, _Counter) for _, _Counter in chunks])
        #     ipca.partial_fit(mat)

        mat = np.array([tfidf(self.meta, _Counter) for _, _Counter in tokenized_news])
        mat_std = StandardScaler().fit_transform(mat)
        pca.fit(mat_std)
        return pca

    def vectorize(self, tokenized_news: list) -> np.array:
        '''
         ニュース原稿のTF-IDFベクトルを求めたのち主成分分析で次元削減する
        '''
        pca = self.incremental_fit(tokenized_news)
        print('IncrementPCA fitting finished')
        data = []
        for category, counter in tokenized_news:
            category_vec = self.meta.category_dic[category]
            vec = tfidf(self.meta, counter).reshape(1, -1)
            # transformの結果は2重リストになっているので、最初の要素を取り出す
            dimred_tfidf = pca.transform(vec)[0]
            # print(dimred_tfidf)
            data.append((category_vec, dimred_tfidf))
        return np.array(data)


def main():

    meta = Metadata()
    meta.build(min_token_len=conf.MINIMUM_TOKEN_LENGTH)
    print('TFIDF calculated')
    print(meta)
    # print(meta.token_to_docid)
    # print(meta.tokenized_news)
    # print(meta.idf)

    pca_tfidf = PcaTfidfVectorizer(meta)

    learning_data = pca_tfidf.vectorize(meta.tokenized_news)
    print('vectorizing finished')

    dirname = str(conf.TFIDF_DIR)
    if not os.path.isdir(dirname):
        os.mkdir(dirname)

    utils.pickle_dump(meta, dirname + '/vector-meta.pkl')
    print('Meta data was dumped.')

    utils.pickle_dump(learning_data, dirname + '/vector-data.pkl')
    print('Learning data was dumped.')
if __name__ == '__main__':
    main()