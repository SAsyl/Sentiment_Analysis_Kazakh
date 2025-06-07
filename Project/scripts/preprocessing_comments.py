from kaznlp.normalization.ininorm import Normalizer
from kaznlp.tokenization.tokrex import TokenizeRex
from kaznlp.normalization.emojiresolver import EmojiResolver

from nltk.stem import SnowballStemmer

from collections import defaultdict

from configs import translated_comments_path
from configs import punctuations
from configs import stopwords_kz_path, stopwords_kz_translit_path, stopwords_ru_path

import pandas as pd
import ast
import string
from tqdm import tqdm
import time


def remove_emojis(line):
    emojirez = EmojiResolver()
    removed_emojis = emojirez.replace(txt=line, replaced_element='empty')

    return removed_emojis


def remove_punctuation_and_stopwords(tokens):
    snowball = SnowballStemmer(language='russian')

    removed_punctuation = [token for token in tokens if token not in punctuations]

    with open(stopwords_kz_translit_path, 'r') as file:
        stopwords_kz = [line.strip() for line in file]

    with open(stopwords_ru_path, 'r') as file:
        stopwords_ru = [line.strip() for line in file]

    stemmed = [snowball.stem(i) for i in removed_punctuation]
    removed_stopwords_kz = [i for i in stemmed if i not in stopwords_kz]
    removed_stopwords_kz_ru = [i for i in removed_stopwords_kz if i not in stopwords_ru]
    filtered = [token for token in removed_stopwords_kz_ru if len(token) >= 2]

    return filtered


def comment_nomalizer(line):
    comment = str(line)
    normalizer = Normalizer()

    normalized = normalizer.normalize(comment, translit=True, desegment=2, dedupe=2, emojiresolve=True, stats=False)
    normalized_lower = normalized.lower()

    normalized_removed_emojis = remove_emojis(normalized_lower)

    return normalized_removed_emojis


def word_tokenize(line):
    tokrex = TokenizeRex()

    tokenized = tokrex.tokenize(line)[0]

    return tokenized


# def joiner(lst):
#     list_from_string = ast.literal_eval(lst)
#     result = ' '.join(list_from_string)
#
#     return result


def preprocessing(df):
    df['normalized'] = df['text'].apply(comment_nomalizer)
    df['tokenized'] = df['normalized'].apply(word_tokenize)

    df['stemmed_lemmed'] = df['tokenized'].apply(remove_punctuation_and_stopwords)

    # df['text_lemm'] = df['stemmed_lemmed'].apply(lambda x: joiner(x))
    df['text_lemm'] = df['stemmed_lemmed'].apply(lambda x: ' '.join(x))

    # print(df.loc[:, ['text', 'normalized', 'label']].values)
    # print(df)

    # df = df.dropna()

    df = df[df['stemmed_lemmed'].apply(lambda x: x != [])]
    df = df[df['text_lemm'].apply(lambda x: x != '')]
    # df = df.dropna()
    df = df.drop_duplicates(subset=['stemmed_lemmed'])
    df = df.dropna(subset=['text', 'label', 'normalized', 'tokenized', 'stemmed_lemmed', 'text_lemm'])
    # print(df)
    # print(df)

    return df


def most_frequently_words(dataset, top=10, bottom=10):
    word_freq = defaultdict(int)
    for tokens in dataset:
        for token in tokens:
            word_freq[token] += 1

    print(f"Unique words: {len(word_freq)}")

    sorted_word_freq = sorted(word_freq, key=word_freq.get, reverse=True)

    topn_frequented = sorted_word_freq[:top]
    lastn_frequented = sorted_word_freq[-bottom:]

    print(f"{top} most popular words: {topn_frequented}\n{top} most popular words: {[word_freq.get(el) for el in topn_frequented]}\n")
    print(f"{top} most unpopular words: {lastn_frequented}\n{top} most unpopular words: {[word_freq.get(el) for el in lastn_frequented]}")
    # print(f"{top} most popular words: {topn_frequented}\n{bottom} most unpopular words: {lastn_frequented}")

    # return topn_frequented, lastn_frequented


def main():
    # comment_filename = sys.argv[1]
    comment_filename = translated_comments_path
    df = pd.read_csv(comment_filename)

    print('--------------------')
    print(f'Shape is {df.shape}')
    print('--------------------')

    # df1000 = df.sample(n=1000, ignore_index=True)
    df1000 = df
    print(f"Распределение классов: positive ({df1000['label'].sum() / df1000.shape[0]}) and negative ({(1 - df1000['label'].sum() / df1000.shape[0])})")
    # print(df1000)

    start_time = time.time()
    df1000 = preprocessing(df1000)
    print(f"--- {(time.time() - start_time) / 60} minutes ---")

    print(df1000.info())

    # most_frequently_words(df1000['stemmed_lemmed'], top=10, bottom=10)

    output_filename = f"{comment_filename[:-4]}_normalized.csv"
    with open(output_filename, 'w', encoding='utf-8') as normalized_file:
        df1000.to_csv(normalized_file, index=False)


if __name__ == '__main__':
    main()