import pandas as pd
import re
import numpy as np

from configs import Qasqa_Jol_links, Azattyq_links


class Extractor:
    def __init__(self, filename=Qasqa_Jol_links):
        self.df = None
        self.filename = filename

    def set_filename(self, filename):
        self.filename = filename

    def read_file(self):
        df = pd.read_csv(self.filename, delimiter=';', header=None)

        df.columns = ["link", "title"]
        df["link"] = df["link"].apply(str)
        df["link"] = df["link"].apply(self.del_num_links)
        df = df[df["link"] != ""]

        self.df = df

    def df_info(self):
        print(f"Number of links: {self.df.shape[0]}")
        # print(self.df)

    def get_links(self):
        return np.array(self.df["link"])

    def del_num_links(self, string):
        new_string = re.sub(r'\d+\. ', '', string)
        return new_string

    def get_link_from_string(self, string):
        youtube_link = ""
        # match = re.search(r'https?://www\.youtube\.com/watch\?v=[\w]+', string)
        match = re.search(r'https?://\S+', string)
        if match:
            youtube_link = match.group()

        return youtube_link


def main():
    # qasqa = Extractor(Azattyq_links)
    # qasqa = Extractor(Qasqa_Jol_links)
    # qasqa.read_file()
    #
    # qasqa.df_info()
    #
    # links = qasqa.get_links()
    # print(links)
    pass


if __name__ == '__main__':
    main()