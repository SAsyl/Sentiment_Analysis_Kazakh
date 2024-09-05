## Этапы создания:
### Что можно сделать:
1. Перевести прошлые датасеты с ENG / RUS на казахский ✅
2. Сделать идентичную маркировку для тональности: ✅
	- Negative -> 0
	- Positive  -> 1
1. Собрать еще 8.000 комментов с различных видео материалов (~ 120 ссылок на видео)
2. Провести качественный сбор: предобработать данные ✅
3. После разметки создать облако позитивных/негативных слов ✅
4. Применить различные подходы к разметке собранных комментариев из YouTube
5. Конфиги для пайплайна

### Preprocessing pipeline:
1. Удаление стопслов ✅
2. Нормализация (лематизация) ✅
3. Токенизация ✅
4. Векторизация ✅
### Сбор данных:
1. Write a YouTube parser comments ✅
2. Find 10 videos (with more than 5 comments) & save their URLs ✅
3. Using a parser to extract all comments ✅
4. Organize a database ✅

### Подходы к разметке данных:
1. Использовать кластеризацию для разметки данных.
2. Использовать перенос обучения предыдущих моделей: BERT, FastText для попытки разметки данных.
3. Разметить 10% -- 20% данных, а после разметить на размеченном.

### Literature:
1. Altynbek Amiruly Sharipbay, Banu Yergesh, Gulmira Bekmanova. "Sentiment analysis of Kazakh text and their polarity (2019)"
2. Tom De Smedt, Guy De Pauw, Pieter Van Ostaeyen. "Automatic Detection of Online Jihadist Hate Speech: CLiPS Technical Report Series (2018)"
	- The work describes a system that detects forbidden statements with an accuracy of more than 80%, using natural language processing techniques and machine learning
3. B. Myrzakhmetov, Zh. Yessenbayev and A. Makazhanov. "Initial Normalization of User Generated Content: Case Study in a Multilingual Setting" to appear in Proceedings of AICT 2018
	- Initial normalization module [kaznlp ](https://github.com/nlacslab/kaznlp)
4. A. Toleu, G. Tolegen, A. Makazhanov. "Character-based Deep Learning Models for Token and Sentence Segmentation" Proceedings of TurkLang 2017
	- Tokenizers _kaznlp_
5.  Habr:
	- [Определение токсичных комментариев на русском языке](https://habr.com/ru/companies/vk/articles/526268/)
	- [Toxic Comments Detection in Russian](https://habr.com/ru/companies/vk/articles/526282/)
	- -----
	- [Автоматическое определение эмоций в текстовых беседах с использованием нейронных сетей](https://habr.com/ru/companies/vk/articles/463045/)
	- [Contextual Emotion Detection in Textual Conversations Using Neural Networks](https://habr.com/ru/companies/vk/articles/439850/)
	- -----
	- ОБЯЗАТЕЛЬНО ПРИ НАПИСАНИИ ТЕКСТА: [Автоматическое определение тональности текста (Sentiment Analysis)](https://habr.com/ru/articles/263171/)
	- [Анализ тональности текстов с помощью сверточных нейронных сетей](https://habr.com/ru/companies/vk/articles/417767/)
	- [4 метода векторизации текстов](https://medium.com/@bigdataschool/4-%D0%BC%D0%B5%D1%82%D0%BE%D0%B4%D0%B0-%D0%B2%D0%B5%D0%BA%D1%82%D0%BE%D1%80%D0%B8%D0%B7%D0%B0%D1%86%D0%B8%D0%B8-%D1%82%D0%B5%D0%BA%D1%81%D1%82%D0%BE%D0%B2-f8ac90e4175a)
6. YouTube:
	- [Как эффективно проводить эксперименты](https://www.youtube.com/watch?v=RS_U6qodpsc):
		a) как выбрать паплайн
		б) различные способы регуляризации
		в) 
7. [10 Sentiment Analysis Project Ideas with Source Code (2024)](https://www.projectpro.io/article/sentiment-analysis-project-ideas-with-source-code/518)
8. [Text Classification Algorithms and Models](https://www.projectpro.io/article/machine-learning-nlp-text-classification-algorithms-and-models/523)
9. ОБЯЗАТЕЛЬНО ПРИ НАПИСАНИИ ТЕКСТА: [Семантический анализ для автоматической обработки естественного языка](https://rdc.grfc.ru/2021/09/semantic_analysis/):
	a) термины и определения
	б) автоматическая обработка текста: pipeline
	в) анализ тональности
	г) архитектуры нейронных сетей для решения задач NLP
	д) приложения и модели для обработки текста
	е) проблемы обработки естественного языка
	 - доступ к данным
	 - открытость данных исследований
	 - определение тональности, иронии и сарказма
	 ж) тренды 2021
10. [Сравнительный анализ тональности комментариев в YouTube](https://youtu.be/aoZ-YEzXZvs):
```diff
+	а) анализ тональности комментариев в YouTube с помощью машинного обучения
+	б) парсер комментариев YouTube
+	в) получать частотность слов в наборах текстов
+	г) создавать красивые графики "облака тэгов"
@@	д) получать векторные представления текстов bag of words и TF-IDF
-	е) классифицировать комментарии с помощью логистической регрессии
-	ж) оценивать качество классификации с помощью графиков ROC-кривых и матрицы ошибок
-	з) визуализировать наиболее важные для классификации слова
-	и) применять полученный классификатор для анализа тональности комментариев.
```
### Log:
#### **Spring 2024**
##### **01.05.2024**:
Написал 2 класса Extractor, HTML_Extractor для извлечения HTML кода YouTube страниц и последующего сохранения. Таким образом сохранил 20+21 кодов страниц.

Еще написал класс Comment_Extractor, позволяющий извлекать комменты из страниц и записывать в общий файл с комментами. Итого собрал 2019 комментов (без обработки).
##### **02.05.2024**:
Попытался сделать перевод прошлых датасетов на казахский, но не получилось. Т.к. стоит ограничение на кол-во переводов в день.
##### **03.05.2024**:
Написал парсер, который автоматические переводит список комментариев из предыдующих датасетов на казахский язык и сохраняет в соответствующий файл. Перевел только 2.000 комментов из _pikabu_2ch_eng.csv_.

Совсем забыл, что в Google Translator можно переводить сразу файлы. Поэтому разделил файлы на 5 малых частей каждый и перевел. Осталось объединить их и провести предобработку.
##### **04.05.2024**:
Объединил разделенные файлы в один, после собрал все переведенные комменты в одном файле _comments_kz.csv_ весом 77 MB. Итого получилось 279.246 комментов:
- labeled_tweets_kz: 226.834,
- pikabu_2ch_kz: 14.412,
- yelp_polarity_kz: 38.000.

Провел несколько этапов предобработки данных с помощью _kaznlp_.
##### **05.05.2024**:
Провел еще несколько этапов предобработки данных с помощью _kaznlp_ и добавил вывод часто встречающихся по частоте (и нечасто встречающихся) слов в датасете.
##### **06.05.2024**:
Ничего не делал. 😓
##### **07.05.2024**:
Ничего не делал. 😓
##### **08.05.2024**:
Ничего не делал. 😓
##### **09.05.2024**:
Сделал визуализацию частотных слов уникалиных для датасетов положительных и отрицательных комментариев.

Построил простую модель _Логистической регрессии_ и вывел значимые слова для соответствующих датасетов.
##### **10.05.2024**:
Начал клепать презентацию с полученными "выводами".
#### **Autumn 2024**:
##### **05.09.2024**:
Нашел статьи для прочтения и добавил их в лог файл в репозиторий гита:
1. [KazSAnDRA: Kazakh Sentiment Analysis Dataset of Reviews and Attitudes](https://arxiv.org/abs/2403.19335)
2. [Assembling the Kazakh Language Corpus](https://aclanthology.org/D13-1104.pdf)
3. [Cross-lingual Language Model Pretraining](https://arxiv.org/pdf/1901.07291)
4. Лемматизатор для казахского языка [Text Normalization and Spelling Correction in Kazakh
Language](https://ceur-ws.org/Vol-2268/paper25.pdf)
5. Normalization of KazNLP [Normalization of Kazakh Texts](https://aclanthology.org/R19-2001.pdf)
##### **06.09.2024**:
❗❗❗ Прочитать статью 1 и 4 ❗❗❗


