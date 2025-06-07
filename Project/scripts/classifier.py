import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, auc, accuracy_score

from configs import translated_comments_path

from visualizer import plot_word_clouds

import plotly.graph_objects as go
from plotly.subplots import make_subplots


class Classifier:
    def __init__(self, df):
        self._df = df
        self._vertorize_type = 'wordbug'

    def fit(self):
        pass

    def predict(self):
        pass

    def vectorize(self, type):
        if type == 'wordbug':
            pass
        elif type == 'tfidf':
            pass
        elif type == 'word2vec':
            pass


def tfidf_distribution(df, plot_filename, text_mark='text', label_mark='label', test_size=0.2, random_state=42, plot=False):
    train, test = train_test_split(df, test_size=test_size, random_state=42)
    # print(train)
    # print(test)

    # Class distribution in train and test sets
    for sample in [train, test]:
        print(sample[sample[label_mark] == 1].shape[0] / sample.shape[0])

    count_idf_positive = TfidfVectorizer(ngram_range=(1, 1))
    count_idf_negative = TfidfVectorizer(ngram_range=(1, 1))

    tf_idf_positive = count_idf_positive.fit_transform(train.query('label == 1')[text_mark])
    tf_idf_negative = count_idf_negative.fit_transform(train.query('label == 0')[text_mark])

    positive_importance = pd.DataFrame(
        {'word': count_idf_positive.get_feature_names_out(),
         'idf': count_idf_positive.idf_
         }).sort_values(by='idf', ascending=False)

    negative_importance = pd.DataFrame(
        {'word': count_idf_negative.get_feature_names_out(),
         'idf': count_idf_negative.idf_
         }).sort_values(by='idf', ascending=False)

    # print(positive_importance.query('word not in @negative_importance.word and idf < 10'))
    # print(negative_importance.query('word not in @positive_importance.word and idf < 10'))

    if plot:
        fig = plt.figure(figsize=(12, 5))
        positive_importance.idf.hist(bins=100,
                                     label='positive',
                                     alpha=0.5,
                                     color='b',
                                     )
        negative_importance.idf.hist(bins=100,
                                     label='negative',
                                     alpha=0.5,
                                     color='r',
                                     )
        plt.title('Distribution of bigrams by TF-IDF values')
        plt.xlabel('TF-IDF')
        plt.ylabel('Word count')
        plt.legend()
        plt.savefig(f"./../results/classifier/tfidf_distibution_{plot_filename}_test-size({test_size}).png")

    return train, test


def plot_ROC_AUC(test, predict_lr_proba, plot_filename):
    fpr_base, tpr_base, _ = roc_curve(test['label'], predict_lr_proba[:, 1])
    roc_auc = auc(fpr_base, tpr_base)

    fig = make_subplots(1, 1,
                        subplot_titles=[f"ROC curve (area = {roc_auc:.3f})"],
                        x_title="False Positive Rate",
                        y_title="True Positive Rate"
                        )

    fig.add_trace(go.Scatter(
        x=fpr_base,
        y=tpr_base,
        # fill = 'tozeroy',
        name="ROC base (area = %0.3f)" % roc_auc,
    ))

    # fig.update_layout(
    #     height=600,
    #     width=800,
    #     xaxis_showgrid=False,
    #     xaxis_zeroline=False,
    #     template='plotly_dark',
    #     font_color='rgba(212, 210, 210, 1)'
    # )

    fig.show()


def plot_confusion_matrix(dataset, predict_proba, threshold, plot_filename, save_plot):
    matrix = confusion_matrix(dataset['label'],
                              (predict_proba[:, 0] < threshold).astype('float'),
                              normalize='true',
                              )

    plt.figure(figsize=(5, 5))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="Blues")
    plt.title(f"Confusion Matrix. Threshold = {threshold:.3f}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    if save_plot:
        plt.show()
    else:
        plt.savefig(f"./../results/classifier/confusion_matrix_{plot_filename}_{threshold:.3f}.png")


def accuracy(predict_proba, threshold, test_df_label):
    pred = (predict_proba[:, 0] < threshold).astype('float')
    return accuracy_score(test_df_label, pred), (predict_proba[:, 0] < threshold).astype('float').mean()


def main():
    df = pd.read_csv(translated_comments_path)
    print(df['label'].head())

    # classifier = Classifier(df)

    train, test = tfidf_distribution(df=df, text_mark='text', label_mark='label', test_size=0.2, plot=False)

    counter_idf = TfidfVectorizer(ngram_range=(1, 1))

    # Получаем словарь и idf только из тренировочного набора данных
    count_train = counter_idf.fit_transform(train['text_lemm'])

    # Применяем обученный векторайзер к тестовому набору данных
    count_test = counter_idf.transform(test['text_lemm'])

    # Инициализируем модель с параметрами по умолчанию
    model_lr = LogisticRegression(max_iter=10000, n_jobs=-1)

    # Подбираем веса для слов с помощь fit на тренировочном наборе данных
    model_lr.fit(count_train, train['label'])

    # Получаем прогноз модели на тестовом наборе данных
    predict_count_proba_train = model_lr.predict_proba(count_train)
    predict_count_proba_test = model_lr.predict_proba(count_test)

    # Объединим в таблицу словарь из нашего векторайзера
    # и веса для слов из обученной модели
    weights = pd.DataFrame({'words': counter_idf.get_feature_names_out(),
                            'weights': model_lr.coef_.flatten()})

    # Создаем копию отсортированную по возрастанию
    weights_min = weights.sort_values(by='weights')
    weights_min['weights'] = weights_min['weights'] * (-1)

    # И еще одну отсортированную по убыванию
    weights_max = weights.sort_values(by='weights', ascending=False)

    # print(weights_min)
    # print(weights_max)

    # plot_word_clouds(pos_df=weights_max,
    #                  neg_df=weights_min,
    #                  n_top_words=100,
    #                  type='classifier')

    # ------------ Find optimal threshold ------------
    # scores = {}
    #
    # weight = 0.6
    #
    # optimal_threshold = None
    # optimal_score = float('-inf')
    #
    # for threshold in np.linspace(0, 1, 100):
    #     matrix = confusion_matrix(test['label'],
    #                               (predict_count_proba_test[:, 0] < threshold).astype('float'),
    #                               normalize='true')
    #
    #     tp = matrix[1, 1]
    #     tn = matrix[0, 0]
    #
    #     if tp > 0.7 and tn > 0.7:
    #         score = tp * weight + tn * (1 - weight)
    #         if score > optimal_score:
    #             optimal_threshold = threshold
    #             optimal_score = score
    #
    # print("Optimal Threshold:", optimal_threshold)
    # print("Optimal Score:", optimal_score)
    # ------------------------------------------------

    # plot_confusion_matrix(test, predict_count_proba_test, optimal_threshold)
    # plot_confusion_matrix(test, predict_count_proba_test, 0.6)

    # plot_ROC_AUC(test, predict_count_proba_test)

    thres = 0.67
    print(f"\nAccuracy on train (threshold = {thres}): {accuracy(predict_count_proba_train, thres, train['label'])}")
    print(f"Accuracy on test (threshold = {thres}): {accuracy(predict_count_proba_test, thres, test['label'])}")


if __name__ == '__main__':
    main()
