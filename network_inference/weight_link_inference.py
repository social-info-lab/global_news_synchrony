''''''

# as described in https://www.sbert.net/docs/pretrained_models.html, the model can be chosen from:
# (1) LaBSE
# (2) paraphrase-multilingual-mpnet-base-v2
# (3) distiluse-base-multilingual-cased-v2

''' sbatch -o script_output/weight_link_inference/weight_link_inference_overall_LaBSE_positive_4_56.out weight_link_inference_script.sh overall LaBSE positive 4 56'''


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
from utils import TEXT_MAX_LEN, truncatetext, score_normalization



'''to do'''
# 5-fold cross validation
# filter less than 10 tokens text



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-lo", "--loss", dest="loss",
                            default="overall", type=str,
                            help="loss type for training. chosen from overall, multilabel1(add 2 aspects: NET and NAR), and multilabel2(add all 6 aspects)")
    parser.add_argument("-m", "--model-name", dest="model_name",
                            default="LaBSE", type=str,
                            help="model name. chosen from LaBSE, paraphrase-multilingual-mpnet-base-v2, and distiluse-base-multilingual-cased-v2")
    parser.add_argument("-nt", "--norm-type", dest="norm_type",
                            default="positive", type=str,
                            help="normalization function type. chosen from positive and unsigned")
    parser.add_argument("-bs", "--batch-size", dest="batch_size",
                            default=4, type=int,
                            help="batch size. maybe should no greater than 8")
    parser.add_argument("-tl", "--tail-length", dest="tail_length",
                            default=0, type=int,
                            help="tail length.")
    args = parser.parse_args()


    '''test'''

    # You can specify any huggingface/transformers pre-trained model here, for example,
    # paraphrase-multilingual-mpnet-base-v2
    # LaBSE
    # distiluse-base-multilingual-cased-v1

    # model_name = "LaBSE"
    # model_name = "paraphrase-multilingual-mpnet-base-v2"
    # model_name = "distiluse-base-multilingual-cased-v2"
    model_name = args.model_name
    main_weight1 = 0.9
    main_weight2 = 0.75

    obj = args.loss
    norm_type = args.norm_type
    train_batch_size = args.batch_size
    tail_used_length = args.tail_length
    method = f'{obj}-{norm_type}-tail{tail_used_length}-batch{train_batch_size}'

    # load data for reproduce of hfl team
    # data = pd.read_csv(f'/Users/xichen/Documents/GitHub/mediacloud/network_inference/truncated_data/df_per_annotation_for_model-tail56.csv')
    #
    # data["text1"] = data.apply(lambda x: str(x["content.title1"])+str(x["content.body1"]), axis=1)
    # data["text2"] = data.apply(lambda x: str(x["content.title2"])+str(x["content.body2"]), axis=1)
    #
    # data = data[["content.pair_id", "real_lang1", "real_lang2", "text1", "text2", "GEO", "ENT", "TIME", "NAR", "overall", "STYLE", "TONE"]]
    # data.rename(columns={"content.pair_id": 'pair_id'}, inplace=True)
    # data.rename(columns={"real_lang1": 'lang1'}, inplace=True)
    # data.rename(columns={"real_lang2": 'lang2'}, inplace=True)
    #
    # data.rename(columns={"GEO": 'Geography'}, inplace=True)
    # data.rename(columns={"ENT": 'Entities'}, inplace=True)
    # data.rename(columns={"TIME": 'Time'}, inplace=True)
    # data.rename(columns={"NAR": 'Narrative'}, inplace=True)
    # data.rename(columns={"overall": 'Overall'}, inplace=True)
    # data.rename(columns={"STYLE": 'Style'}, inplace=True)
    # data.rename(columns={"TONE": 'Tone'}, inplace=True)
    #
    # full_training_set = data[:int(0.8*len(data))]
    # full_testing_set = data[int(0.8*len(data)):]
    # full_training_set.to_csv("/Users/xichen/Downloads/full_training_set.csv")
    # full_testing_set.to_csv("/Users/xichen/Downloads/full_testing_set.csv")


    '''loading data'''
    data = pd.read_csv(f'truncated_data/df_per_annotation_for_model-tail{tail_used_length}.csv')
    data.rename(columns={'content.pair_id': 'pair_id'}, inplace=True)
    data = data.dropna(subset=['OverallNorm', 'content.title1', 'content.title2', 'content.body1', 'content.body2'])
    data["OverallNorm"] = data.apply(lambda x: math.pow(x["OverallNorm"], 3), axis=1)

    # remove the article with less than 10 tokens
    data["body1_len"] = data.apply(lambda x: len(x['content.body1'].strip().split(' ')), axis=1)
    data["body2_len"] = data.apply(lambda x: len(x['content.body2'].strip().split(' ')), axis=1)
    data = data[data["body1_len"] >= 10]
    data = data[data["body2_len"] >= 10]

    '''for comparing to semeval results'''
    # # (1) using original dataset
    # semeval_train_data = pd.read_csv(f'../semeval_baseline/train_data_for_baseline.csv')
    # data = pd.merge(data, semeval_train_data, how='left', on='pair_id')
    # data = data.dropna(subset=['Overall'])
    #
    # synthetic_df = pd.DataFrame(data.groupby(["pair_id"])["Overall"].mean())
    #
    # data.drop(['Overall'], axis=1, inplace = True)
    # data = pd.merge(data, synthetic_df, how='left', on='pair_id')
    # data.drop_duplicates(subset=['pair_id'], keep ='first', inplace = True)

    # (2) using all dataset
    # semeval_test_data = pd.read_csv(f'../semeval_baseline/test_data_for_baseline.csv')
    # test_take_out = semeval_test_data['pair_id'].to_list()
    # # data = data[~data["pair_id"].isin(test_take_out)]
    # train = data[~data["pair_id"].isin(test_take_out)]
    # dev = data[data["pair_id"].isin(test_take_out)]
    # test = data[data["pair_id"].isin(test_take_out)]



    train_test_split_ratio = 0.8
    real_train_dev_split_ratio = 0.9
    # train_test_split_ratio = 1
    # real_train_dev_split_ratio = 0.95
    train_test_split_pos = int(train_test_split_ratio * len(data))
    train = data[:train_test_split_pos]
    train = train.sample(frac=1).reset_index(drop=True)
    real_train_dev_split_pos = int(real_train_dev_split_ratio * len(train)) # shuffle the training and dev set
    real_train = train[:real_train_dev_split_pos]
    dev = train[real_train_dev_split_pos:]

    test = data[(train_test_split_pos-1):]

    train_samples = []
    for item in train.itertuples():
        if obj == "overall":
            origin_score = item[14]
        elif obj == "multilabel1":
            # multi-label objective for training
            origin_score = main_weight1 * item[14] + (1 - main_weight1)/2 * (item[17] + item[19])
        else:
            origin_score = main_weight2 * item[14] + (1 - main_weight2)/6 * (item[16] + item[17] + item[18] + item[19] + item[20] + item[21])
        score = score_normalization(norm_type, origin_score)
        url1_text = item[7] + " " + item[9]
        url2_text = item[8] + " " + item[10]


        inp_example = InputExample(texts=[url1_text, url2_text], label=score)
        train_samples.append(inp_example)

    model_save_path = f"model/{model_name}/{method}"
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    train_start_time = datetime.now()
    print("training starts at: ", train_start_time.strftime("%Y-%m-%d_%H-%M-%S"))

    device = "cuda:0"

    # Load a pre-trained sentence transformer model
    model = SentenceTransformer(model_name, device=device)
    model.max_seq_length = TEXT_MAX_LEN

    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)
    train_loss = losses.CosineSimilarityLoss(model=model)

    dev_samples = []
    for item in dev.itertuples():
        origin_score = item[14]

        score = score_normalization(norm_type, origin_score)
        url1_text = item[7] + " " + item[9]
        url2_text = item[8] + " " + item[10]


        inp_example = InputExample(texts=[url1_text, url2_text], label=score)
        dev_samples.append(inp_example)

    test_samples = []
    for item in test.itertuples():
        origin_score = item[14]

        score = score_normalization(norm_type, origin_score)
        url1_text = item[7] + " " + item[9]
        url2_text = item[8] + " " + item[10]


        inp_example = InputExample(texts=[url1_text, url2_text], label=score)
        test_samples.append(inp_example)

    dev_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, name='sts-test')

    print("\n clean memory...")
    gc.collect()
    torch.cuda.empty_cache()

    model.fit(train_objectives=[(train_dataloader, train_loss)],
              epochs=4,
              evaluator=dev_evaluator,
              output_path=model_save_path,
              optimizer_params={'lr': 2e-5},
              use_amp=True)

    train_end_time = datetime.now()
    print("training ends at: ", train_end_time.strftime("%Y-%m-%d_%H-%M-%S"))
    print(f"training takes {train_end_time-train_start_time} seconds in total...")

    infer_start_time = datetime.now()
    print("inference starts at: ", infer_start_time.strftime("%Y-%m-%d_%H-%M-%S"))

    model = SentenceTransformer(model_save_path)
    test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, name='sts-test')

    infer_end_time = datetime.now()
    print(f"inference takes {infer_end_time-infer_start_time} seconds in total...")

    print(test_evaluator(model, output_path=model_save_path))