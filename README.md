[//]: # (# Sentiment analysis for chats in Kazakh language)
# Анализ тональности чатов на казахском языке

## Описание

Данная работа посвящена анализу тональности текстов на казахском языке. В ходе работы были выполнены следующие этапы:

1. Сбор данных с открытых датасетов на русском и английском языках, а также с новостных сайтов и комментариев к видео материалам с платформы YouTube.
2. Перевод собранных данных на казахский язык.
3. Предобработка данных, включающая нормализацию, токенизацию и удаление пунктуации и знаков препинания.
4. Построение облаков слов на сырых и предобработанных данных.
5. Построение модели на основе Логистической регрессии.
6. Получение метрик модели, таких как матрица ошибок и ROC-кривая.

## Цель

Цель данной работы - разработка модели для автоматического анализа тональности текстов на казахском языке с использованием современных методов машинного обучения.

## Критерии оценивания модели

Для оценки качества модели использовались следующие метрики:

- **Accuracy (Точность)**: доля правильно классифицированных объектов среди всех объектов.
- **ROC AUC (Площадь под ROC-кривой)**: вероятность того, что случайно выбранный положительный пример будет иметь более высокую оценку, чем случайно выбранный отрицательный пример.

## Использование

### Предварительные требования

- Python 3.x

[//]: # (- Jupyter Notebook)
- Установленные библиотеки: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`

### Запуск

1. Клонируйте репозиторий:
    ```sh
    git clone https://github.com/SAsyl/Sentiment_Analysis_Kazakh.git
    cd Sentiment_Analysis_Kazakh
    ```

## Заключение и дальнейшая работа

В заключении к работе планируется выполнить следующие этапы:

1. Написание алгоритма для приведения слов к начальной форме (для казахского языка).
2. Векторизация данных, используя embedding.
3. Использование трансформера для построения модели.

## Контрибьюторы

- [Shakhputov Assylkhan](https://github.com/SAsyl)

[//]: # (## Лицензия)

[//]: # ()
[//]: # (Этот проект лицензируется на условиях лицензии MIT. Подробности см. в файле [LICENSE]&#40;LICENSE&#41;.)

# Дополнительные ссылки
- Ссылка на [презентацию](https://drive.google.com/file/d/1495mbekmbvRf5bAg0dREpa95mopnw7Tf/view?usp=sharing).
- Ссылка на [датасет](https://drive.google.com/file/d/1OBuYdmCz5Ru5JBD8_CdHYeR195DpeM5M/view?usp=sharing).