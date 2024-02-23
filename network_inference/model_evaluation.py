''''''


# as described in https://www.sbert.net/docs/pretrained_models.html, the model can be chosen from:
# (1) LaBSE
# (2) paraphrase-multilingual-mpnet-base-v2
# (3) distiluse-base-multilingual-cased-v2


''' sbatch -o script_output/model_evaluation/model_evaluation_overall_LaBSE_positive_4_56.out model_evaluation_script.sh overall LaBSE positive 4 56'''


import torch, gc
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, LoggingHandler, losses, util, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers import SentenceTransformer, SentencesDataset, LoggingHandler, losses
from sentence_transformers.readers import InputExample
from argparse import ArgumentParser
import math
import logging
from datetime import datetime
import os
import gzip
import csv
import copy
import json
from collections import Counter
import pandas as pd
import re
import jieba
from utils import TEXT_MAX_LEN, truncatetext, score_normalization, score_reverse_normalization
from scipy.stats import pearsonr
import numpy as np


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument("-lo", "--loss", dest="loss",
                            default="overall", type=str,
                            help="loss type for training. chosen from overall, multi_label1(add 2 aspects: NET and NAR), and multi_label2(add all 6 aspects)")
    parser.add_argument("-m", "--model-name", dest="model_name",
                            default="LaBSE", type=str,
                            help="model name.")
    parser.add_argument("-nt", "--norm-type", dest="norm_type",
                        default="positive", type=str,
                        help="normalization function type.")
    parser.add_argument("-bs", "--batch-size", dest="batch_size",
                        default=4, type=int,
                        help="batch size.")
    parser.add_argument("-tl", "--tail-length", dest="tail_length",
                        default=0, type=int,
                        help="tail length.")
    args = parser.parse_args()


    obj = args.loss
    model_name = args.model_name
    norm_type = args.norm_type
    train_batch_size = args.batch_size
    tail_used_length = args.tail_length
    method = f'{obj}-{norm_type}-tail{tail_used_length}-batch{train_batch_size}'
    # temporary method route
    # method = f'{obj}-{norm_type}-tail{tail_used_length}-batch{train_batch_size}-09'


    # # baseline
    # model_save_path = "LaBSE"
    model_save_path = f"model/{model_name}/{method}"

    '''loading model'''
    device = "cuda:0"
    model = SentenceTransformer(model_save_path)

    '''loading data'''
    train_test_split_ratio = 0.8

    data = pd.read_csv(f'truncated_data/df_per_annotation_for_model-tail{tail_used_length}.csv')
    data.rename(columns={'content.pair_id': 'pair_id'}, inplace=True)
    data = data.dropna(subset=['OverallNorm', 'content.title1', 'content.title2', 'content.body1', 'content.body2'])
    data = data[int(train_test_split_ratio*len(data)):]

    # temp for comparing cluster similarity
    data = data[data['OverallNorm'] == 0]
    # data = data[data['OverallNorm'] < 0.3]
    # data = data[data['OverallNorm'] > 0]


    '''for comparing to semeval results'''
    # semeval_test_data = pd.read_csv(f'../semeval_baseline/test_data_for_baseline.csv')
    # data = pd.merge(data, semeval_test_data, how='left', on='pair_id')
    # data = data.dropna(subset=['Overall'])
    #
    # synthetic_df = pd.DataFrame(data.groupby(["pair_id"])["Overall"].mean())
    #
    # # # per item form
    # # data.drop(['Overall'], axis=1, inplace = True)
    # # data = pd.merge(data, synthetic_df, how='left', on='pair_id')
    # # data.drop_duplicates(subset=['pair_id'], keep ='first', inplace = True)
    #
    # # per annotation form
    # synthetic_df.drop(['Overall'], axis=1, inplace = True)
    # data = pd.merge(data, synthetic_df, how='left', on='pair_id')


    '''for test'''
    # data = data[:10]

    start_time = datetime.now()
    print("data processing start at: ", start_time.strftime("%Y-%m-%d_%H-%M-%S"))

    item_count = 0
    data_samples = []
    predict_score_samples = []
    origin_score_samples = []
    for item in data.itertuples():
        item_count += 1
        if item_count % 100 == 0:
            print(f"processed {item_count} pairs...  ", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

        # ne_sim = item[3]
        # score = item[14]
        origin_score = item[14]
        score = score_normalization(norm_type, origin_score)
        url1_text = item[7] + " " + item[9]
        url2_text = item[8] + " " + item[10]

        emb1 = model.encode(url1_text)
        emb2 = model.encode(url2_text)
        cos_sim = util.pytorch_cos_sim(emb1, emb2)
        predict_score = score_reverse_normalization(norm_type, cos_sim)

        # constrain the boundary
        if predict_score > 4:
            predict_score = 4
        if predict_score < 1:
            predict_score = 1

        inp_example = InputExample(texts=[url1_text, url2_text], label=score)
        data_samples.append(inp_example)
        predict_score_samples.append(predict_score)
        origin_score_samples.append(origin_score)


    end_time = datetime.now()
    print("data processing ends at: ", end_time.strftime("%Y-%m-%d_%H-%M-%S"))
    print(f"data processing {end_time - start_time} seconds in total...")

    '''STS benchmark'''
    test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(data_samples, name='sts-test')
    # print("STS benchmark: ", test_evaluator(model, output_path=model_save_path))
    print("STS benchmark: ", test_evaluator(model))

    '''pearson correlation'''
    print("pearson corrleation: ",pearsonr(predict_score_samples, origin_score_samples))

    '''average dissimilarity'''
    print("average dissimilarity", np.mean(predict_score_samples))
