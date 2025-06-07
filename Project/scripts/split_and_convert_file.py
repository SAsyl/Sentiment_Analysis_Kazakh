import os
import pandas as pd
import numpy as np

from configs import pikabu_2ch_eng, yelp_polarity_eng, labeled_tweets_rus


def main():
    # ------ Split the file into parts and save them in '.csv' format ------
    filename = labeled_tweets_rus
    df = pd.read_csv(filename)

    parts = 1
    if filename in {yelp_polarity_eng, labeled_tweets_rus}:
        parts = 5

    splits = np.array_split(df, parts)

    for i, split in enumerate(splits):
        split.to_csv(f'{filename[:-4]}_{i + 1}.csv', index=False)

    # ------------------------------------------------------

    # ------ Convert files from '.csv' format to '.xlsx' -------
    for i in range(parts):
        fname = f"{filename[:-4]}_{i + 1}.csv"
        df = pd.read_csv(fname)

        outname = f"{fname[:-4]}.xlsx"
        df.to_excel(outname, index=False)
        # print(outname)
    # -------------------------------------------------------


if __name__ == '__main__':
    main()