import pandas as pd
import numpy as np
import os


def main():
    filename = './../Data/tweets_2ch_labeled/pikabu_2ch_kz.csv'
    df = pd.read_csv(filename)

    df['label'] = df['label'].replace({0: 1, 1: 0})

    df.to_csv(filename, index=False)


if __name__ == '__main__':
    main()