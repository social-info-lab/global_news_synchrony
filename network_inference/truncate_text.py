import pandas as pd
from utils import truncatetext
import os
from argparse import ArgumentParser
from datetime import datetime


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-tl", "--tail-length", dest="tail_length",
                            default=0, type=int,
                            help="tail length.")
    args = parser.parse_args()

    text_max_length = 512
    tail_used_length = args.tail_length

    LABELS = ["Dont-Use-Zero", "Very Similar", "Somewhat Similar", "Somewhat Dissimilar", "Very Dissimilar", "Other"]
    data = pd.read_csv('df_per_annotation_for_model.csv')


    print("overall score transoformation...", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    data['response'] = data['response'].apply(lambda x: eval(x))
    data['overall'] = data['response'].apply(lambda x: LABELS.index(x[4]))
    data = data.drop(data[data['overall'] == 5].index)
    data['OverallNorm'] = data['overall'].apply(lambda x: (4-x)/3)

    data['GEO'] = data['response'].apply(lambda x: LABELS.index(x[0]))
    data['ENT'] = data['response'].apply(lambda x: LABELS.index(x[1]))
    data['TIME'] = data['response'].apply(lambda x: LABELS.index(x[2]))
    data['NAR'] = data['response'].apply(lambda x: LABELS.index(x[3]))
    data['STYLE'] = data['response'].apply(lambda x: LABELS.index(x[5]))
    data['TONE'] = data['response'].apply(lambda x: LABELS.index(x[6]))

    print("text body transformation...", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    data['content.body1'] = data.apply(lambda x: truncatetext(x['content.body1'], x['real_lang1'], text_max_length, tail_used_length), axis=1)
    data['content.body2'] = data.apply(lambda x: truncatetext(x['content.body2'], x['real_lang2'], text_max_length, tail_used_length), axis=1)

    print("data sampling... ", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    data = data.sample(frac=1).reset_index(drop=True) # shuffle the data
    data = data.dropna(subset=['OverallNorm', 'content.title1', 'content.title2', 'content.body1', 'content.body2'])

    output_dir = f'truncated_data'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    data.to_csv(f"{output_dir}/df_per_annotation_for_model-tail{tail_used_length}.csv")