from preprocessing_comments import comment_nomalizer

with open('./stopwords/stopwords_kz.txt', 'r') as file:
    # Read the file contents and split each line into elements
    my_list = [line.strip() for line in file]

print(my_list)

normalized_list = [comment_nomalizer(word) for word in my_list]
print(normalized_list)

with open('./stopwords/stopwords_kz_transliterated.txt', 'w') as file:
    for word in normalized_list:
        file.write(str(word) + '\n')
