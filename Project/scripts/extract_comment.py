import time
import webbrowser

from bs4 import BeautifulSoup
import lxml
import requests
import os
import csv
import json
from lxml import etree
from lxml import html

from configs import Qasqa_Jol_links, Azattyq_links

all_comments_csv = './../Data/YouTube/parsed/comments.csv'


class Comment_Extractor:
    def __init__(self, filename=Qasqa_Jol_links):
        self.filename = filename
        self.file = filename.split("/")[-2]
        self.source = None
        self.xpath = ('/html/body/ytd-app/div[1]/ytd-page-manager/ytd-watch-flexy/div[5]/div[1]/div/div['
                      '2]/ytd-comments/ytd-item-section-renderer/div[3]')
        self.comment_xpath = 'ytd-comment-thread-renderer'

    def read_file_by_index(self, index):
        with open(f"./../Data/YouTube/parsed/{self.file}/{index}.html", "r", encoding='utf-8') as html_file:
            self.source = html_file.read()

    def extract_comments(self):
        all_comments = []

        # soup = BeautifulSoup(self.source, 'lxml')
        soup = BeautifulSoup(self.source, 'html.parser')

        tree = etree.HTML(str(soup))

        # comment_index = 1
        # # span_text = soup.find('span', xpath=f'/html/body/ytd-app/div[1]/ytd-page-manager/ytd-watch-flexy/div[5]/div[1]/div/div[2]/ytd-comments/ytd-item-section-renderer/div[3]/ytd-comment-thread-renderer[2]/ytd-comment-view-model/div[3]/div[2]/ytd-expander/div/yt-attributed-string')
        # text = tree.xpath('/html/body/ytd-app/div[1]/ytd-page-manager/ytd-watch-flexy/div[5]/div[1]/div/div[2]/ytd-comments/ytd-item-section-renderer/div[3]/ytd-comment-thread-renderer[1]/ytd-comment-view-model/div[3]/div[2]/ytd-expander/div/yt-attributed-string/span')
        comment_blocks = tree.xpath('//*[@id="content-text"]/span')

        for block in comment_blocks:
            text = block.text
            if text is not None:
                all_comments.append(text)

        return all_comments

    def conjoine_comments(self, comments):
        conjoined_comments = []
        comm = ''
        comment_index = 0
        while comment_index < len(comments):
            if comments[comment_index] != '\n':
                comm += comments[comment_index].strip()
                comment_index += 1

                if comment_index < len(comments) and comments[comment_index] != '\n':
                    conjoined_comments.append(comm)
                    comm = ''
            else:
                comm += ' '
                comment_index += 1

        return conjoined_comments

    def save_comments(self, comments, filename):
        # table_header = ["comment", "toxic"]
        #
        # with open(filename, "w", newline='', encoding='utf-8') as file:
        #     writer = csv.writer(file)
        #
        #     writer.writerow((table_header))

        for comment in comments:
            data = [comment, 0]
            with open(filename, "a", newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow((data))


def main():
    # comment_extractor = Comment_Extractor(filename=Qasqa_Jol_links)
    comment_extractor = Comment_Extractor(filename=Azattyq_links)
    # comment_extractor.read_file_by_index(1)
    # comments = comment_extractor.extract_comments()
    # print(comments)

    for i in range(21):
        comment_extractor.read_file_by_index(i + 1)
        comments = comment_extractor.extract_comments()

        conjoined_comments = comment_extractor.conjoine_comments(comments=comments)
        print(f"{i + 1}: {len(conjoined_comments)}")
        # print(conjoined_comments)

        comment_extractor.save_comments(comments=conjoined_comments, filename=all_comments_csv)

    # print((''.join(comments)).split('\n'))

    # dict3 = {}
    #
    # publ_time = article_header.find(class_="uk-text-muted").text.strip()
    # views = article_header.find(class_="arrilot-widget-container uk-text-muted").text.strip()
    # link_name = all_articles[article_link]["link_name"]
    # topic = all_articles[article_link]["topic"]
    #
    # dict3["link_name"] = link_name
    # dict3["views"] = views
    # dict3["publ_time"] = publ_time
    # dict3["topic"] = topic
    #
    # dict_all_articles[article_link] = dict3
    #
    # index += 1
    #
    # with open("data/json/all_articles1.json", "w", encoding='utf-8') as file:
    #     json.dump(dict_all_articles, file, indent=4, ensure_ascii=False)

    # # print(len(dict_all_articles))
    # # for k in dict_all_articles.keys():
    # #     print(k)


if __name__ == '__main__':
    main()