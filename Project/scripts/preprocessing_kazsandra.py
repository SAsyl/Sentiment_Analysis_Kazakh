import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import json

from collections import defaultdict

import ast

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score

from classifier import tfidf_distribution, plot_ROC_AUC, plot_confusion_matrix, accuracy
from visualizer import plot_word_clouds

from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score

import torch

import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Embedding, LSTM, GRU, Dense, Dropout, Bidirectional, GlobalAveragePooling1D
from tensorflow.keras.callbacks import ModelCheckpoint


from kaznlp.normalization.ininorm import Normalizer
from kaznlp.tokenization.tokrex import TokenizeRex
from kaznlp.normalization.emojiresolver import EmojiResolver

import gensim
from gensim.models import Word2Vec
from keras.preprocessing.sequence import pad_sequences

# Limit GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


class Normalizer:
    def __init__(self, file_roots, file_affixes, data_prepared=False):
        self._file_affixes = file_affixes
        self.roots = self._load_roots(file_roots)
        self.all_type_of_affixes = self._get_type_of_affixes(file_affixes)

        self._keyValueEndings = ('caseEnds', 'pluralEnds', 'possessiveEnds', 'personalEnds')
        self._keyValueSuffixes = ('none2adjSuff', 'verb2adjSuff',
                                  'verbSuff1', 'verbSuff2', 'verbSuff3', 'verbSuff4',
                                  'prepSuff', 'pronSuff', 'pastSuff', 'negativeSuff')

        self._data_prepared = data_prepared

    def _load_roots(self, file_roots):
        with open(file_roots, 'r', encoding='utf-8') as file:
            roots = file.read().split('\n')

        roots = list(set(roots))
        roots.remove('')

        return roots

    def _get_type_of_affixes(self, file_affixes):
        with open(file_affixes, 'r', encoding="utf-8") as f:
            all_affixes = json.load(f)

        # ------- Import Endings -------
        self._CaseEndings = set(all_affixes['Endings']['CaseEndings'])
        self._PluralEndings = set(all_affixes['Endings']['PluralEndings'])
        self._PossessiveEndings = set(all_affixes['Endings']['PossessiveEndings'])
        self._PersonalEndings = set(all_affixes['Endings']['PersonalEndings'])

        # Combine all Endings into one Dict
        self._TotalEndings = self._CaseEndings | self._PluralEndings | self._PossessiveEndings | self._PersonalEndings
        # print(len(self._TotalEndings))

        # Sorted by length
        SortedByLenTotalEndings = sorted(self._TotalEndings, key=len, reverse=True)
        # print(len(SortedByLenTotalEndings))

        # ------- Import Suffixes -------
        self._Nouns2AdjSuffixes = set(all_affixes['Suffixes']['Nouns2AdjSuffixes'])

        self._Verbs2AdjSuffixes = set(all_affixes['Suffixes']['Verbs2AdjSuffixes'])

        self._VerbsSuffixes1 = set(all_affixes['Suffixes']['VerbsSuffixes1'])
        self._VerbsSuffixes2 = set(all_affixes['Suffixes']['VerbsSuffixes2'])
        self._VerbsSuffixes3 = set(all_affixes['Suffixes']['VerbsSuffixes3'])
        self._VerbsSuffixes4 = set(all_affixes['Suffixes']['VerbsSuffixes4'])

        self._PrepositionsSuffixes = set(all_affixes['Suffixes']['PrepositionsSuffixes'])

        self._PronounsSuffixes = set(all_affixes['Suffixes']['PronounsSuffixes'])

        self._pastSuffixes = set(all_affixes['Suffixes']['pastSuffixes'])

        self._negativeSuffixes1 = set(all_affixes['Suffixes']['negativeSuffixes1'])
        self._negativeSuffixes2 = set(all_affixes['Suffixes']['negativeSuffixes2'])

        # Combine all Suffixes into one Dict
        self._TotalSuffixes = self._Nouns2AdjSuffixes | self._Verbs2AdjSuffixes | self._VerbsSuffixes1 | self._VerbsSuffixes2 | self._VerbsSuffixes3 | self._VerbsSuffixes4 | self._PrepositionsSuffixes | self._PronounsSuffixes | self._pastSuffixes | self._negativeSuffixes1 | self._negativeSuffixes2
        # print(len(self._TotalSuffixes))

        # Sorted by length
        SortedByLenTotalSuffixes = sorted(self._TotalSuffixes, key=len, reverse=True)
        # print(len(SortedByLenTotalSuffixes))

        typeOfAffixes = {
            'caseEnds': self._CaseEndings,
            'pluralEnds': self._PluralEndings,
            'possessiveEnds': self._PossessiveEndings,
            'personalEnds': self._PersonalEndings,

            'none2adjSuff': self._Nouns2AdjSuffixes,
            'verb2adjSuff': self._Verbs2AdjSuffixes,
            'verbSuff1': self._VerbsSuffixes1,
            'verbSuff2': self._VerbsSuffixes2,
            'verbSuff3': self._VerbsSuffixes3,
            'verbSuff4': self._VerbsSuffixes4,
            'prepSuff': self._PrepositionsSuffixes,
            'pronSuff': self._PronounsSuffixes,
            'pastSuff': self._pastSuffixes,
            'negativeSuff': self._negativeSuffixes1 | self._negativeSuffixes2
        }

        return typeOfAffixes

    def normalize(self, data: pd.DataFrame, text_mark='text', label_mark='label'):
        # negative - 0
        # positive - 1

        if not self._data_prepared:
            data['normalized'] = data[text_mark].apply(str.lower)
            data['tokenized'] = data['normalized'].apply(lambda x: x.split(' '))
            data['stemmed_lemmed'] = data['tokenized'].apply(lambda tokens: self.lemmatize(tokens))
            data['text_lemm'] = data['stemmed_lemmed'].apply(lambda x: ' '.join([el for el in x[0] if el != '']))
            data['all_affixes'] = data['stemmed_lemmed'].apply(lambda x: self._combine_all_affixes(x[1]))
            data['stemmed_lemmed'] = data['text_lemm'].apply(lambda x: x.split(' '))

            self._data_prepared = True


    def _combine_all_affixes(self, affixes):
        concatenated = np.concatenate(affixes)
        return concatenated


    def _remove_affixes(self, word, affixes, wordAffixes, typeOfAffixes):
        AtLeastOneAffix = False
        affixes = sorted(affixes, key=len, reverse=True)
        DeleteSuffix = False

        if word.endswith('у'):
            wordAffixes.append('y')
            word = word[:-1]

        if word in self.roots:
            return word, wordAffixes[::-1]

        for affix in affixes:
            if word.endswith(affix):
                # print(word, affix)

                len_affix = len(affix)
                wordAffixes.append(affix)

                AtLeastOneAffix = True

                for key, dictAffixes in typeOfAffixes.items():
                    if affix in dictAffixes:
                        affixes = set(affixes) - dictAffixes
                        typeOfAffixes.pop(key)
                        # print(key, affix)

                        if key in self._keyValueSuffixes:
                            DeleteSuffix = True

                            for keyEnds in self._keyValueEndings:
                                try:
                                    typeOfAffixes.pop(keyEnds)
                                except KeyError:
                                    continue

                        # print(key)
                        # print()

                        break

                break

        if AtLeastOneAffix == False:
            return word, wordAffixes[::-1]

        # print(word, wordAffixes)
        return self._remove_affixes(word[:-len_affix], affixes, wordAffixes, typeOfAffixes)

    def lemmatize(self, tokens):
        total_sentence = []
        all_affixes = []
        for i, token in enumerate(tokens):
            type_affixes = self._get_type_of_affixes(self._file_affixes)
            prediction_word, word_affixes = self._remove_affixes(token,
                                                                 self._TotalEndings | self._TotalSuffixes,
                                                                 [],
                                                                 type_affixes)
            total_sentence.append(prediction_word)
            all_affixes.append(word_affixes)
        return [total_sentence, all_affixes]


    def plot_lengths_of_roots_distribution(self, upper_limit=None):
        roots_corpus = pd.DataFrame(self.roots, columns=['Root'])
        roots_corpus['Length'] = roots_corpus['Root'].apply(len)

        max_len = max(roots_corpus['Length']) if upper_limit is None else upper_limit
        min_len = min(roots_corpus['Length'])

        # print(f"Max length: {max_len}")
        # print(f"Min length: {min_len}")

        plt.figure(figsize=(8, 6))

        n, bins, patches = plt.hist(roots_corpus['Length'], bins=range(1, max_len), edgecolor='black')

        # Add the bin values to the plot
        for i in range(len(patches)):
            x = (patches[i].get_x() + patches[i].get_width() / 2)
            y = patches[i].get_height()
            plt.text(x, y, str(int(y)), ha='center', va='bottom')

        plt.xticks(range(1, max_len + 1))

        plt.title(f'Distribution of roots length. Min: {min_len}, Max: {max_len}')
        plt.show()

        return roots_corpus

    def most_frequently_words(self, dataset, top=10, bottom=10):
        word_freq = defaultdict(int)
        for tokens in dataset:
            for token in tokens:
                word_freq[token] += 1

        print(f"Unique words: {len(word_freq)}")

        sorted_word_freq = sorted(word_freq, key=word_freq.get, reverse=True)

        top_N_frequented = sorted_word_freq[:top]
        last_N_frequented = sorted_word_freq[-bottom:]

        print(
            f"{top} most popular words: {top_N_frequented}\n{top}Frequency: {[word_freq.get(el) for el in top_N_frequented]}\n")
        print(
            f"{top} most unpopular words: {last_N_frequented}\nFrequency: {[word_freq.get(el) for el in last_N_frequented]}")
        # print(f"{top} most popular words: {top_N_frequented}\n{bottom} most unpopular words: {last_N_frequented}")

        # return top_N_frequented, last_N_frequented

def plot_word_clouds_by_weights(weights, plot_filename, n_top=100, type='classifier'):
    # Создаем копию отсортированную по возрастанию
    weights_min = weights.sort_values(by='weights')
    weights_min['weights'] = weights_min['weights'] * (-1)

    # И еще одну отсортированную по убыванию
    weights_max = weights.sort_values(by='weights', ascending=False)

    plot_word_clouds(pos_df=weights_max,
                     neg_df=weights_min,
                     n_top_words=n_top,
                     type=type,
                     result_filename=plot_filename)

def LogRegModel(max_iter, n_jobs, tfidf_counter, train, test):
    X_train = tfidf_counter.transform(train['text_lemm'])
    y_train = train['label']
    print(f'Count train shape: {X_train.shape}')

    # Применяем обученный векторайзер к тестовому набору данных
    X_test = tfidf_counter.transform(test['text_lemm'])
    y_test = test['label']
    print(f'Count test shape: {X_test.shape}')

    # Инициализируем модель с параметрами по умолчанию
    model = LogisticRegression(max_iter=max_iter, n_jobs=n_jobs)

    # Подбираем веса для слов с помощь fit на тренировочном наборе данных
    model.fit(X_train, y_train)

    # Объединим в таблицу словарь из нашего векторайзера и веса для слов из обученной модели
    weights = pd.DataFrame({'words': tfidf_counter.get_feature_names_out(),
                            'weights': model.coef_.flatten()})

    return model, weights

def plot_AUC_curve(X_train, y_train, X_test, y_test, model, plot_filename):
    for name, X, y, model in [
        ('train', X_train, y_train, model),
        ('test ', X_test, y_test, model)
    ]:
        proba = model.predict(X)                    # [:, 0]
        # proba = model.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, proba)
        plt.plot(*roc_curve(y, proba)[:2], label='%s AUC=%.4f' % (name, auc))

    plt.plot([0, 1], [0, 1], '--', color='black', )
    plt.legend(fontsize='large')
    plt.grid()

    if plot_filename is not None:
        plt.savefig(f'./../results/classifier/AUC_curve_{plot_filename}.png')
    else:
        plt.show()

    print('Train accuracy', accuracy_score((model.predict(X_train) > 0.55).astype(int), y_train))
    print('Test accuracy', accuracy_score((model.predict(X_test) > 0.55).astype(int), y_test))

def preprocess_data(input_filename, output_filename, data_prepared=True):
    file_roots = './../Data/roots.txt'
    file_affixes = './../Data/affixes.json'

    # ----------------- Preprocessing Data --------------------
    # Applying the rule-based normalizer
    kazakh_normalizer = Normalizer(file_roots, file_affixes, data_prepared=data_prepared)

    # ----- Столбчатая диаграмма распределения длин корней ----
    # kazakh_normalizer.plot_lengths_of_roots_distribution(upper_limit=13)
    # ---------------------------------------------------------

    if not data_prepared:
        df = pd.read_csv(input_filename)  # Not normalized

        # print(df)

        # print(sum(df['label']) / df.shape[0])


        kazakh_normalizer.normalize(data=df, text_mark='text_cleaned', label_mark='label')
        # print(train_text_label)

        df.to_csv(output_filename, index=False)
    else:
        df = pd.read_csv(output_filename)
        df['stemmed_lemmed'] = df['stemmed_lemmed'].apply(lambda x: ast.literal_eval(x))

    print(
        f"Распределение классов: positive ({df['label'].sum() / df.shape[0]}) and negative ({(1 - df['label'].sum() / df.shape[0])})")
    # ---------------------------------------------------------
    df.dropna(inplace=True)

    return df

def decode_review(text, reverse_word_index):
    return ' '.join([reverse_word_index.get(word, '?') for word in text])

def encode_review(text, word_index):
    words = str(text).lower().split()
    words = ['<START>'] + words
    idxs = [word_index.get(word, word_index['<UNKNOWN>']) for word in words]
    return idxs

def w2v_BiLSTM_model(model_w2v, embed_dim, seq_len):
    model = Sequential()

    # Добавление слоя эмбеддинга с инициализацией весами Word2Vec
    model.add(Embedding(input_dim=model_w2v.wv.vectors.shape[0],
                        output_dim=embed_dim,
                        weights=[model_w2v.wv.vectors],
                        input_length=seq_len,
                        trainable=False))  # Для заморозки весов

    model.add(Bidirectional(LSTM(128, return_sequences=True, dropout=0.5)))
    model.add(Bidirectional(LSTM(64, return_sequences=True, dropout=0.5)))
    model.add(Bidirectional(LSTM(32, return_sequences=False, dropout=0.4)))
    model.add(Dense(32, activation='relu'))
    # model.add(Bidirectional(LSTM(16, return_sequences=False, dropout=0.3, recurrent_dropout=0.3)))
    model.add(Dense(1, activation='sigmoid'))

    return model

def w2v_Dense_model(model_w2v, embed_dim, seq_len):
    model = tf.keras.Sequential()
    model.add(Embedding(input_dim=model_w2v.wv.vectors.shape[0],
                        output_dim=embed_dim,
                        weights=[model_w2v.wv.vectors],
                        input_length=seq_len,
                        trainable=False))

    model.add(GlobalAveragePooling1D())
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    return model

def plot_loss_accuracy(history):
    epochs = range(1, len(history.history['accuracy']) + 1)

    plt.figure()
    plt.plot(epochs, history.history['loss'], 'bo', label='Training loss')
    plt.plot(epochs, history.history['val_loss'], 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure()
    plt.plot(epochs, history.history['accuracy'], 'bo', label='Training accuracy')
    plt.plot(epochs, history.history['val_accuracy'], 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()
    plt.show()

def get_dict_of_words_W2V(model_w2v_filename):
    model = Word2Vec.load(model_w2v_filename)
    # print(model.wv.vectors.shape)

    word_index = {k: (v + 3) for k, v in model.wv.key_to_index.items()}
    word_index["<PAD>"] = 0
    word_index["<START>"] = 1
    word_index["<UNKNOWN>"] = 2
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

    return model, word_index, reverse_word_index

def main():
    train = preprocess_data(input_filename='./../Data/KazSAnDRA/02_pc_train_ros.csv',
                            output_filename='./../Data/results/train_(all_dataset)_v1.0.csv',
                            data_prepared=True)

    valid = preprocess_data(input_filename='./../Data/KazSAnDRA/04_pc_valid.csv',
                            output_filename='./../Data/results/valid_(all_dataset)_v1.0.csv',
                            data_prepared=True)

    test = preprocess_data(input_filename='./../Data/KazSAnDRA/05_pc_test.csv',
                           output_filename='./../Data/results/test_(all_dataset)_v1.0.csv',
                           data_prepared=True)

    # # -------------- Print most N frequent words --------------
    # kazakh_normalizer.most_frequently_words(train_df['stemmed_lemmed'], 20, 20)
    # # ---------------------------------------------------------


    # # Let's take the random oversampling dataset
    # train_text_label = train_df.loc[:, ['text_lemm', 'label']]  # Not normalized
    # print(train_text_label)
    #
    # plot_filename = 'kazsandra_(uni)gram'
    #
    # train, valid = tfidf_distribution(df=train_text_label, plot_filename=plot_filename,
    #                                   text_mark='text_lemm', label_mark='label',
    #                                   test_size=0.2, random_state=42, plot=False)



    # # -------------------- TF-IDF on lemmatized text --------------------
    # counter_idf = TfidfVectorizer(ngram_range=(1, 1), max_features=1000)
    # X_train = counter_idf.fit_transform(train['text_lemm']).toarray()
    # y_train = train['label']
    # # # print([(el, i) for (i, el) in enumerate(X_train[0]) if el != 0.0])
    # # # print([(el, i) for (i, el) in enumerate(X_train[-1]) if el != 0.0])
    # X_val = counter_idf.transform(valid['text_lemm']).toarray()
    # y_val = valid['label']
    #
    # X_test = counter_idf.transform(test['text_lemm']).toarray()
    # y_test = test['label']
    # # --------------------------------------------------------------------

    # -------------------- Word2Vec on lemmatized text --------------------
    # w2v_model_100 = Word2Vec(
    #     sentences=train['stemmed_lemmed'],
    #     vector_size=100,
    #     window=3,
    #     min_count=1,
    #     workers=4,
    #     sg=1
    # )
    # w2v_model_100.save("./../models/Word2Vec/v1.0.model")
    model, word_index, reverse_word_index = get_dict_of_words_W2V("./../models/Word2Vec/v1.0.model")

    # text = 'Салам Асыл'
    #
    # print(encode_review(text, word_index))
    # print(decode_review(encode_review(text, word_index), reverse_word_index))

    for dataset in [train, valid, test]:
        dataset['sequence'] = dataset['text_lemm'].apply(lambda x: encode_review(x, word_index))
    # print(train.head())

    pos_ex_1 = 'Бұл өнім өте жақсы көрінеді.'

    MAX_SEQ_LEN = 1000
    EMBEDDING_DIM = model.vector_size
    print(
        f"MAX_SEQUENCE_LENGTH: {max(train['sequence'].apply(len))} or {MAX_SEQ_LEN}, Embedding dim: {EMBEDDING_DIM}")

    X_train = pad_sequences(train['sequence'], maxlen=MAX_SEQ_LEN, padding='post')
    y_train = train['label'].values

    X_val = pad_sequences(valid['sequence'], maxlen=MAX_SEQ_LEN, padding='post')
    y_val = valid['label'].values

    X_test = pad_sequences(test['sequence'], maxlen=MAX_SEQ_LEN, padding='post')
    y_test = test['label'].values

    print(model.wv.vectors.shape[0], EMBEDDING_DIM, model.wv.vectors.shape, MAX_SEQ_LEN)


    # model_clf = w2v_BiLSTM_model(model_w2v=model, embed_dim=EMBEDDING_DIM, seq_len=MAX_SEQ_LEN)

    # model_clf = w2v_Dense_model(model_w2v=model, embed_dim=EMBEDDING_DIM, seq_len=MAX_SEQ_LEN)
    # model_clf = load_model('./../models/kazsandra_w2v/v4_0.1782.keras')

    model_clf = load_model('./../models/kazsandra_1gram/v1_0.8286.keras')

    print(model_clf.summary())

    # BATCH_SIZE = 16
    # NUM_EPOCHS = 10
    #
    # # Компиляция модели
    # model_clf.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    #
    # # Обучение модели
    # history = model_clf.fit(X_train, y_train,
    #                         epochs=NUM_EPOCHS,
    #                         batch_size=BATCH_SIZE,
    #                         steps_per_epoch=100,
    #                         validation_data=(X_val, y_val),
    #                         verbose=1)
    #
    # plot_loss_accuracy(history)


    # model = Sequential([
    #     Dense(1000, activation='relu', input_shape=(X_train.shape[1],)),  # Input layer
    #     Dropout(0.5),  # Dropout for regularization
    #     Dense(500, activation='relu'),  # Hidden layer
    #     Dropout(0.5),  # Dropout for regularization
    #     Dense(250, activation='relu'),  # Hidden layer
    #     Dropout(0.3),  # Dropout for regularization
    #     Dense(120, activation='relu'),  # Hidden layer
    #     Dropout(0.3),  # Dropout for regularization
    #     Dense(50, activation='relu'),  # Hidden layer
    #     Dense(1, activation='sigmoid')  # Output layer (softmax for multi-class classification)
    # ])
    #
    # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # model.fit(X_train, y_train, epochs=10, batch_size=16, validation_data=(X_val, y_val))
    #
    # Step 7: Evaluate the model
    loss, accuracy = model_clf.evaluate(X_test, y_test)
    print(f"Test Accuracy: {accuracy:.4f}")
    plot_filename = f'kazsandra_w2v_v4_{accuracy:.4f}'

    # # Step 8: Make predictions
    # predictions = model_clf.predict(X_test)
    # predicted_labels = (predictions > 0.55).astype(int)  # Convert probabilities to labels
    # print("Predicted Labels:", predicted_labels)

    # Save the model
    # model_clf.save(f'./../models/kazsandra_w2v/v3_{accuracy:.4f}.keras')  # Saves the model to a file
    # print("Model saved successfully!")

    # print(plot_filename)
    # model_lr, weights_lr = LogRegModel(max_iter=10000, n_jobs=-1, tfidf_counter=counter_idf,
    #                                    train=train, test=valid)

    # plot_word_clouds_by_weights(weights=weights_lr, plot_filename=plot_filename, n_top=100)
    #
    plot_AUC_curve(X_train, y_train, X_test, y_test,
                   model=model_clf,
                   plot_filename=None) # plot_filename
    #
    #
    # # thres = 0.67
    # # for thres in [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]:
    # #     plot_confusion_matrix(test, predict_count_proba_test, thres, plot_filename=plot_filename, save_plot=False)
    # #     print(f"\nAccuracy on train (threshold = {thres}): {accuracy(predict_count_proba_train, thres, train['label'])}")
    # #     print(f"Accuracy on test (threshold = {thres}): {accuracy(predict_count_proba_test, thres, test['label'])}")





if __name__ == '__main__':
    main()
