import nltk
from nltk.corpus import stopwords

# nltk.download('punkt')
# nltk.download('stopwords')

# print(len(stopwords.words('english')), stopwords.words('english'))
# print(len(stopwords.words('russian')), stopwords.words('russian'))
# print(len(stopwords.words('kazakh')), stopwords.words('kazakh'))


def main():
    with open('stopwords_kz.txt', 'w') as file:
        for word in stopwords.words('kazakh'):
            file.write(str(word) + '\n')


if __name__ == '__main__':
    main()
