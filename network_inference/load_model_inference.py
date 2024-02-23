''''''

# sbatch -o script_output/load_model_inference/load_model_inference_170_180.out load_model_inference_script.sh "../ner_art_sampling/network_pairs/candidates/candidates-top10-ne-filtered_170_180/*" 170 180 overall positive 4 56


# languages in a pair are saved as language1 and a2_language by typo...



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
import glob
from utils import TEXT_MAX_LEN, truncatetext, load_article, score_reverse_normalization


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument("-i", "--input-glob", dest="input_glob",
                            default="../ner_art_sampling/network_pairs/candidates/candidates-top10-ne-filtered_170_180/*", type=str,
                            help="Input globstring.")
    parser.add_argument("-s", "--start-date", dest="start_date",
                            default=0, type=int,
                            help="The start date for this ne-art index.")
    parser.add_argument("-e", "--end-date", dest="end_date",
                            default=180, type=int,
                            help="The end date for this ne-art index.")
    parser.add_argument("-lo", "--loss", dest="loss",
                        default="overall", type=str,
                        help="loss type for training. chosen from overall, multi_label1(add 2 aspects: NET and NAR), and multi_label2(add all 6 aspects)")
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


    model_save_path = f"model/LaBSE/{args.loss}-{args.norm_type}-tail{args.tail_length}-batch{args.batch_size}"
    model = SentenceTransformer(model_save_path)
    data_path = args.input_glob


    n_processedpairs = 0
    n_succeedpairs = 0
    n_failedpairs = 0
    start_time = datetime.now()

    files=list(glob.glob(data_path))
    for file in files:
        output_file = file.replace("candidates","prediction")
        output_dir = "/".join(output_file.split("/")[:-1])
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        embed_file = file.replace("candidates", "embedding")
        embed_dir = "/".join(embed_file.split("/")[:-1])
        if not os.path.exists(embed_dir):
            os.makedirs(embed_dir)

        with open(output_file, "w") as outfile:
            with open(embed_file, "w") as embfile:
                with open(file, "r") as fh:
                    for line in fh:
                        n_processedpairs += 1
                        if n_processedpairs % 1000 == 0:
                            print(f"Already processed {n_processedpairs} pairs...  ", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), flush=True)

                        '''for test'''
                        # if n_processedpairs > 10000:
                        #     break

                        try:
                            pair = json.loads(line)
                            a1_file = pair["article1"][0]
                            a1_lineno = pair["article1"][1]
                            a1_date = pair["article1"][2]
                            a2_file = pair["article2"][0]
                            a2_lineno = pair["article2"][1]
                            a2_date = pair["article2"][2]



                            art1 = load_article(a1_file, a1_lineno)
                            art2 = load_article(a2_file, a2_lineno)

                            a1_text_input = art1["title"] + " " + truncatetext(art1["story_text"], art1["language"], TEXT_MAX_LEN, args.tail_length)
                            a2_text_input = art2["title"] + " " + truncatetext(art2["story_text"], art2["language"], TEXT_MAX_LEN, args.tail_length)

                            a1_emb = model.encode(a1_text_input)
                            a2_emb = model.encode(a2_text_input)

                            a1_emb_list = a1_emb.tolist()
                            a2_emb_list = a2_emb.tolist()

                            cos_sim = util.pytorch_cos_sim(a1_emb, a2_emb)
                            predict_score = score_reverse_normalization(args.norm_type, cos_sim)

                            # take care!! this is language1 and a2_language by typo!!
                            predict_pair = {
                                "similarity": predict_score,
                                "date1":a1_date,
                                "stories_id1":art1["stories_id"],
                                "language1":art1["language"],
                                "media_id1":art1["media_id"],
                                "media_name1":art1["media_name"],
                                "media_url1": art1["media_url"],
                                "date2": a2_date,
                                "stories_id2": art2["stories_id"],
                                "a2_language": art2["language"],
                                "media_id2": art2["media_id"],
                                "media_name2": art2["media_name"],
                                "media_url2": art2["media_url"],
                            }

                            outfile.write(json.dumps(predict_pair))
                            outfile.write("\n")



                            emb_pair = {
                                "embedding1": a1_emb_list,
                                "embedding2": a2_emb_list,
                            }

                            embfile.write(json.dumps(emb_pair))
                            embfile.write("\n")

                        except:
                            n_failedpairs += 1
                        n_succeedpairs += 1

    end_time = datetime.now()
    print(f"Finished at {end_time:%Y-%m-%d %H:%M:%S}")
    print(f"Took {end_time - start_time} seconds to processing {n_processedpairs} pairs....")
    print(f"{n_succeedpairs} succeeded pairs and {n_failedpairs} failed pairs....")
