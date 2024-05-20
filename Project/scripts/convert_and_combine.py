import pandas as pd
import glob

from configs import pikabu_2ch_eng, yelp_polarity_eng, labeled_tweets_rus


def combine_main_files(total_filename):
    # Get a list of all XLSX files in the directory
    csv_files = glob.glob(f'./../Data/tweets_2ch_labeled/*.csv')

    # Initialize an empty list to store DataFrames
    dfs = []

    # Load each XLSX file into a separate DataFrame and store it in the list
    for file in csv_files:
        df = pd.read_csv(file, header=0)
        dfs.append(df)

    # Combine all DataFrames into one large DataFrame
    combined_df = pd.concat(dfs, ignore_index=True)

    print(combined_df)

    # Save the combined DataFrame to a CSV file
    combined_df.to_csv(total_filename, index=False)


def combine_separate_files(filename):
    # Get a list of all XLSX files in the directory
    xlsx_files = glob.glob(f'{filename[:-8]}_kz_translated_*')

    # Initialize an empty list to store DataFrames
    dfs = []

    # Load each XLSX file into a separate DataFrame and store it in the list
    for file in xlsx_files:
        df = pd.read_excel(file, header=0)
        dfs.append(df)

    # Combine all DataFrames into one large DataFrame
    combined_df = pd.concat(dfs, ignore_index=True)

    if filename == yelp_polarity_eng:
        combined_df.columns = ['label', 'text']
        combined_df['label'] = combined_df['label'] - 1

        combined_df = combined_df[['text', 'label']]
    elif filename in {labeled_tweets_rus, pikabu_2ch_eng}:
        combined_df.columns = ['text', 'label']

    print(combined_df)

    # Save the combined DataFrame to a CSV file
    combined_df.to_csv(f"./../Data/tweets_2ch_labeled/{filename.split('/')[-2]}_kz.csv", index=False)


def main():
    # combine_separate_files(filename=pikabu_2ch_eng)
    combine_main_files(total_filename='./../Data/tweets_2ch_labeled/comments_kz.csv')


if __name__ == '__main__':
    main()