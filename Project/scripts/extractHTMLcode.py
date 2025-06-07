import time

from selenium import webdriver

from extract_links import Extractor

import pandas as pd
import re
import numpy as np

from configs import Qasqa_Jol_links, Azattyq_links


class HTML_Extractor:
    def __init__(self, filename=Qasqa_Jol_links, delay=30):
        self.filename = filename
        self.delay = delay

        extr = Extractor(filename)
        extr.read_file()

        links = extr.get_links()
        self.links = links
        self.file = self.filename.split("/")[-2]

    def extract_html(self):
        driver = webdriver.Chrome()

        index_link = 1

        for link in self.links:
            driver.get(url=link)

            time.sleep(self.delay)

            with open(f"./../Data/YouTube/parsed/{self.file}/{index_link}.html", "w", encoding='utf-8') as file:
                file.write(driver.page_source)

            index_link += 1

        driver.quit()


def main():
    # ------------ Review only 1 url ------------
    # # HTML code extraction
    # index = 2
    # url = "https://www.youtube.com/watch?v=6g9wPm98J2o"
    #
    # driver.get(url=url)
    #
    # time.sleep(20)
    # # ------------ Save html page --------
    # with open(f"./../Data/YouTube/parsed/{index}.html", "w", encoding='utf-8') as file:
    #     file.write(driver.page_source)
    # # ------------------------------------

    # ------------------------------------------

    # ------------ List of url -----------------

    qasqa_html_extractor = HTML_Extractor(filename=Azattyq_links, delay=30)
    qasqa_html_extractor.extract_html()

    # azat_html_extractor = HTML_Extractor(Azattyq_links, 20)
    # azat_html_extractor.extract_html()

    # ------------------------------------------
    # comment block class = "yt-core-attributed-string yt-core-attributed-string--white-space-pre-wrap"

    # headers = {
    #     # "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)
    #     # Chrome/110.0.0.0 Safari/537.36"
    #     "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 "
    #                   "Safari/537.36"
    # }


if __name__ == '__main__':
    main()