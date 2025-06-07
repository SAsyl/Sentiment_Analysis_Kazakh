import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from configs import labeled_tweets_rus

from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud


def plot_word_clouds(pos_df, neg_df, n_top_words, type, result_filename):
    pos_df = pos_df.head(n_top_words)
    neg_df = neg_df.head(n_top_words)

    # Генерируем красивые картинки со словами на которых размер шрифта соответствует частотности
    wordcloud_positive = WordCloud(background_color="black",
                                   colormap='Blues_r',
                                   max_words=200,
                                   mask=None,
                                   width=1600,
                                   height=1600).generate_from_frequencies(dict(pos_df.values))

    wordcloud_negative = WordCloud(background_color="black",
                                   colormap='Oranges_r',
                                   max_words=200,
                                   mask=None,
                                   width=1600,
                                   height=1600).generate_from_frequencies(dict(neg_df.values))

    # Выводим картинки сгенерированные вордклаудом с помощью
    fig, ax = plt.subplots(1, 2, figsize=(20, 12))

    ax[0].imshow(wordcloud_positive, interpolation='bilinear')
    ax[1].imshow(wordcloud_negative, interpolation='bilinear')

    ax[0].set_title('positive', fontsize=20)
    ax[1].set_title('negative', fontsize=20)

    ax[0].axis("off")
    ax[1].axis("off")

    if type == 'wordcloud':
        plt.savefig(f"./../results/wordcloud/{result_filename}_pos_neg_raw{n_top_words}.png")
    elif type == 'classifier':
        plt.savefig(f"./../results/classifier/{result_filename}_pos_neg_lemmatized{n_top_words}.png")
    elif type == 'tfidf_embed':
        plt.savefig(f"./../results/embedding/tf_idf/{result_filename}_pos_neg_lemmatized{n_top_words}.png")


def create_cloud_positive_and_negative(df, n_top_words):
    positive_comments = df[df['label'] == 1]
    negative_comments = df[df['label'] == 0]

    positive_counter = CountVectorizer(ngram_range=(1, 1))
    negative_counter = CountVectorizer(ngram_range=(1, 1))

    # Получаем словарь уникальных слов (fit) и сразу же считаем частотность для каждого текста (transform)
    positive_count = positive_counter.fit_transform(positive_comments['text_lemm'])
    negative_count = negative_counter.fit_transform(negative_comments['text_lemm'])

    # print("Positive counter: ", positive_count.sum(axis=0).shape[1])
    # print("Negative counter: ", negative_count.sum(axis=0).shape[1])

    # Получаем словарь из CountVectorizer c помощью .get_feature_names_out()
    positive_frequence = pd.DataFrame(
        {'word': positive_counter.get_feature_names_out(),
         'frequency': np.array(positive_count.sum(axis=0))[0]
         }).sort_values(by='frequency', ascending=False)

    # print("Positive frequence:\n", positive_frequence)

    negative_frequence = pd.DataFrame(
        {'word': negative_counter.get_feature_names_out(),
         'frequency': np.array(negative_count.sum(axis=0))[0]
         }).sort_values(by='frequency', ascending=False)

    # print("Negative frequence:\n", negative_frequence)

    # Убираем с помощью запроса пересекающиеся слова и оставляем 100 наиболее частотных
    negative_frequence_filtered = negative_frequence.query('word not in @positive_frequence.word')
    positive_frequence_filtered = positive_frequence.query('word not in @negative_frequence.word')

    plot_word_clouds(pos_df=negative_frequence_filtered,
                     neg_df=positive_frequence_filtered,
                     n_top_words=n_top_words,
                     type='wordcloud')


def main():
    comment_filename = normalized_comments_path
    df = pd.read_csv(comment_filename)

    create_cloud_positive_and_negative(df, n_top_words=10)


if __name__ == '__main__':
    main()
