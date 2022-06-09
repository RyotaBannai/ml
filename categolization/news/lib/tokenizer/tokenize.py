import ftplib
import csv
import glob
import posixpath
import re
import os
import tempfile
from ftplib import error_perm
from posixpath import dirname

import MeCab
import mojimoji as mj

from config import conf
from lib.utility import utils

# MAKE_PATH = lambda f: os.path.join(os.path.dirname(__file__), "./corpus/" + f)


class Tokenizer:
    def __init__(self):
        self._m = MeCab.Tagger(' -d  ' + conf.NEOLOGD_DIR)
        # compileしておく
        self.eng_sentences = re.compile(r'[a-zA-Z0-9]+[ ,\.\'\:\;\-\+?\!]')
        self.numbers = re.compile(r'[0-9０-９]+')
        self.symbols1 = re.compile(r'[\!\?\#\$\%\&\'\*\+\-\.\^_\`\|\~\:]+')
        self.symbols2 = re.compile(r'[\<\=\>\;\{\}\[\]\`\@\(\)\,\\]+')
        self.cjk_symbols = re.compile(r'[“└┐（）【】『』｛｝「」［］《》〈〉！？＝]+')

    def sanitize(self, text: str) -> str:
        # 英文を取り除く（日本語の中の英字はそのまま）
        text = re.sub(self.eng_sentences, '', text)
        # 記号や数字は「 」に変換する。
        # (単純に消してしまうと意味不明な長文になりjanomeがエラーを起こす)
        text = re.sub(self.numbers, '0', text)
        text = re.sub(self.symbols1, ' ', text)
        text = re.sub(self.symbols2, ' ', text)
        text = re.sub(self.cjk_symbols, ' ', text)
        return text

    def tokenize(self, text: str) -> list:
        token_list = []
        append = token_list.append
        try:
            text = mj.zen_to_han(text, kana=False)  # 全角英数はすべて半角英数にする
            text = mj.han_to_zen(text, digit=False, ascii=False)  # 半角カタカナはすべて全角にする
            text = text.lower()  # 英語はすべて小文字にする
            tokens = self._m.parse(text).split('\n')
        except IndexError:
            print(text)
            return None
        for token in tokens:
            # 表層形\t品詞,品詞細分類1,品詞細分類2,品詞細分類3,活用形,活用型,原形,読み,発音
            token = re.split(r'[\,\t]', token)
            if len(token) < 10:
                continue
            ps = token[1]
            if ps not in ['名詞', '動詞', '形容詞']:
                continue
            # 原形があれば原形をリストに入れる
            w = token[7]
            if w == '*' or w == '':
                # 原形がなければ表層系(原稿の単語そのまま)をリストに入れる
                w = token[0]
            if w == '' or w == '\n':
                continue
            append(w)
        return token_list


def text_to_token():
    for filename, chunks in all_chunks.items():
        with tempfile.TemporaryFile() as fp:
            for chunk in chunks:
                category = chunk['category']
                ynt = Tokenizer()
                sanitize = ynt.sanitize(chunk['manuscript'])
                tokens = ynt.tokenize(sanitize)
                line = ' '.join(tokens) + '\n'
                fp.write(line.encode('utf-8'))

            filepath = '/{0}/{1}/{2}.token'.format(
                settings.FTP_TOKEN_DIR,
                category,
                filename)
            dirname, filename = posixpath.split(filepath)
            ftp_makedirs_cwd(ftp, dirname)
            fp.seek(0)
            ftp.storbinary('STOR %s' % filepath, fp)
    ftp.quit()


def ftp_makedirs_cwd(ftp, path, first_call=True):
    """Set the current directory of the FTP connection given in the `ftp`
    argument (as a ftplib.FTP object), creating all parent directories if they
    don't exist. The ftplib.FTP object must be already connected and logged in.
    """
    try:
        ftp.cwd(path)
    except error_perm:
        ftp_makedirs_cwd(ftp, dirname(path), False)
        ftp.mkd(path)
        if first_call:
            ftp.cwd(path)


def read_tokens():
    wakati_paths = conf.TOKEN_DIR.glob('**/*.token')
    print('reading token files')
    for path in wakati_paths:
        category = utils.extract_category(path)
        with open(path, 'r') as f:
            all_wakati = f.read().split('\n')
        for line in all_wakati:
            tokens = line.split(',')
            yield (category, tokens)


if __name__ == '__main__':
    t = Tokenizer()
    categories = conf.DATA_DIR.glob('*')
    #for category in conf.CATEGORIES:
    for category in list(categories):
        category_path = category
        category = str(category_path).split('/')[-1]
        category_token_path = str(conf.TOKEN_DIR / category)
        if not os.path.isdir(category_token_path):
            os.makedirs(category_token_path)
        filenames = category_path.glob('*')
        for filename in list(filenames):
            with open(str(filename), 'r') as tmp:
                reader = csv.reader(tmp)
                result = [{
                    'category': rows[0],
                    'title': t.tokenize(rows[1]),
                    'text_len': rows[2],
                    'text': t.tokenize(rows[3])
                } for rows in reader]

            short_filename = (str(filename).split('/')[-1]).split('.')[0]
            with open(category_token_path+'/'+short_filename+'.token', "a+") as file:
                for chunk in result:
                    file.write((",".join(chunk['title']+chunk['text'])+'\n'))
