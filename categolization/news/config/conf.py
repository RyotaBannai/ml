import os
import pathlib
NEOLOGD_DIR = '/usr/local/lib/mecab/dic/mecab-ipadic-neologd'
PACKAGE_DIR = pathlib.Path(os.path.dirname(__file__)).parents[0]
DATA_DIR = PACKAGE_DIR / 'data/csv'
TOKEN_DIR = PACKAGE_DIR / 'data/token/csv'
TFIDF_DIR = PACKAGE_DIR / 'data/vector'
DATA_FILE = PACKAGE_DIR / 'data/vector/tfidf-data.pkl'
METADATA_FILE = PACKAGE_DIR / 'data/vector/tfidf-meta.pkl'
CORE_DIR = PACKAGE_DIR / 'core/params'
LOG_FILE = PACKAGE_DIR / 'log/tmp/yn_categories_logs'

KEEP_PROB = 0.5
NUM_UNITS1 = 64
NUM_UNITS2 = 64
BATCH_SIZE = 50
TOTAL_STEP = 10000
LEARNING_RATE = 0.00001  # 学習率
TRAINING_DATA_RATIO = 0.8  # 全データのうち訓練用に使う割合

#CATEGORIES = ['社会', '国際総合']
MINIMUM_TOKEN_LENGTH = 50
PCA_DIMENSION = 100
PCA_BATCH_DATA_LENGTH = 50

