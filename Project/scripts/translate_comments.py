from translate import Translator
from selenium import webdriver
# from selenium.webdriver.chrome.service import Service
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import pandas as pd
import numpy as np
import os
import csv
import time

from configs import pikabu_2ch_eng, yelp_polarity_eng, labeled_tweets_rus
from configs import translate_linkEN2KZ_link, translate_linkRU2KZ_link
from configs import chrome_driver_path, firefox_driver_path


class Translator2KZ:
    def __init__(self, filename, from_lang):
        self.to_lang = 'kk'
        self.from_lang = from_lang
        self.df = None
        self.filename = filename
        self.translated_file = f'{self.filename[:-8]}_kz.csv'
        self.translator = Translator()

    def read_file(self):
        df = pd.read_csv(self.filename)
        self.df = df

    def translate(self):
        self.df['text_kz'] = self.df['text'].apply(self.translate_comment)

    def partion_translate_and_save(self, from_index, to_index):
        df_part = self.df.loc[from_index:to_index]

        print(f"Translate begin. From {from_index} to {to_index}")
        df_part['text_kz'] = df_part['text'].apply(self.translate_comment)
        print(f"Translate finished.")

        data = df_part[['text_kz', 'label']].values
        print(f"Save translated dataframe to {self.translated_file}")
        for row in data:
            with open(self.translated_file, "a", newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow((row))
        print(f"Save finished.")

    def translate_comment(self, comment):
        try:
            translated = self.translator.translate(comment, src='en', dest='kk')
            return translated.text if translated else None
        except Exception as e:
            print(f"Error translating text: {e}")
            return None

        # # translator_ru_2_kz = Translator(to_lang='kk', from_lang='en')
        # # translator_eng_2_kz = Translator(to_lang='kk', from_lang='ru')
        # comment = comment.strip()
        # translator_lg_2_kz = Translator(to_lang='kk', from_lang=self.from_lang)
        #
        # translated_comment = translator_lg_2_kz.translate(comment)
        #
        # return translated_comment

    def save_translated(self):
        self.df.to_csv(self.translated_file, index=False)


class TranslatorEN2KZ:
    def __init__(self, filename, from_language, delay):
        self.filename = filename
        self.df = None
        self.translated_file = f'{self.filename[:-8]}_kz.csv'
        self.from_lang = from_language
        self.delay = delay

        translate_link = ""
        if self.from_lang == 'ru':
            translate_link = translate_linkRU2KZ_link
        elif self.from_lang == 'en':
            translate_link = translate_linkEN2KZ_link
        self.translate_link = translate_link

        # ser = Service(executable_path="./../chromedriver_linux/chromedriver")
        options = webdriver.ChromeOptions()
        options.add_argument("--headless")
        # driver = webdriver.Chrome(service=ser, options=options)
        driver = webdriver.Chrome(options=options)
        self.driver = driver

    def read_file(self):
        df = pd.read_csv(self.filename)
        self.df = df

    def partion_translate_and_save(self, from_index, to_index):
        self.driver.get(translate_linkEN2KZ_link)

        if to_index > self.df.shape[0]:
            texts_part = self.df.loc[from_index:]['text'].values
            labels_part = self.df.loc[from_index:]['label'].values
        else:
            texts_part = self.df.loc[from_index:to_index]['text'].values
            labels_part = self.df.loc[from_index:to_index]['label'].values

        texts_part = np.array([el.strip() for el in texts_part])

        # print(f"Translate begin. From {from_index} to {to_index}")
        start_time = time.time()
        approximate_end_time = start_time + ((self.delay + 1) * (to_index - from_index))

        print(f"Translate begin at {self._get_time_by_seconds(start_time)}. Approximate completed at {self._get_time_by_seconds(approximate_end_time)} seconds. From {from_index} to {to_index}.")

        translated_texts_part = []
        xpath_target = '/html/body/c-wiz/div/div[2]/c-wiz/div[2]/c-wiz/div[1]/div[2]/div[2]/c-wiz[2]/div/div[6]/div/div[1]/span[1]'

        for comment in texts_part:
            translate_output = "None"

            try:
                translate_input = self.driver.find_element(By.XPATH,
                                                           '/html/body/c-wiz/div/div[2]/c-wiz/div[2]/c-wiz/div[1]/div[2]/div[2]/c-wiz[1]/span/span/div/textarea')
                translate_input.clear()
                translate_input.send_keys(comment)
                time.sleep(self.delay)

                translate_output = self.driver.find_element(By.XPATH, f"{xpath_target}/span/span")
                translate_output = translate_output.text
            except Exception as ex:
                translate_output = "None"
            finally:
                translated_texts_part.append(translate_output)

        translated_texts_part = np.array(translated_texts_part)
        print(f"Translate finished.")
        end_time = time.time()
        execution_time = end_time - start_time

        print(f"Finished at {self._get_time_by_seconds(end_time)}.Time for translation: {execution_time} seconds.")

        data = np.stack((translated_texts_part, labels_part), axis=1)
        print(f"Save translated dataframe to {self.translated_file}")
        for row in data:
            with open(self.translated_file, "a", newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow((row))
        print(f"Save finished.")

    def _get_time_by_seconds(self, seconds):
        time_struct = time.localtime(seconds)
        return time.strftime("%H:%M:%S", time_struct)

    def close_driver(self):
        self.driver.close()
        self.driver.quit()


def main():
    print("Work started!")

    # ser = Service(executable_path=firefox_driver_path)
    # options = webdriver.FirefoxOptions()
    # driver = webdriver.Firefox(service=ser, options=options)
    # driver = webdriver.Firefox()

    translator_from_en_to_kz = TranslatorEN2KZ(filename=pikabu_2ch_eng, from_language='en', delay=2)
    translator_from_en_to_kz.read_file()
    # index = 4
    # start = index * 1000
    # end = start + 1000

    start = 8000
    end = 15000

    translator_from_en_to_kz.partion_translate_and_save(start, end)

    translator_from_en_to_kz.close_driver()

    # translator2kz = Translator2KZ(filename=pikabu_2ch_eng, from_lang='en')
    # translator2kz.read_file()
    # translator2kz.partion_translate_and_save(0, 5)
    # translator2kz.translate()
    # translator2kz.save_translated()

    print("Work finished!")


if __name__ == '__main__':
    main()