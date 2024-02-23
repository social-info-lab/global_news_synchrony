'''script usage example:'''
'''sbatch --output=script_output/ne_art_index/ne_art_index_0_31.txt -e script_output/ne_art_index/ne_art_index_0_31.err ne_art_index_script.sh 0 31'''

from argparse import ArgumentParser
import glob
from collections import defaultdict
import ast
import traceback
import numpy as np

top_k = 10 # only choose the articles sharing at least one name entity among its top_k tf_idf list
lang_list = ['en','de','es','pl','zh','fr','ar','tr','it','ru']

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-i", "--index-filename", dest="index_filename",
                            default="indexes/*-wiki-v2-filtered.index", type=str,
                            help="Index filename.")
    parser.add_argument("-o", "--ouput-filename", dest="output_fileprefix",
                            default="indexes/ne_art_indexes/top10-ne-art-wiki-filtered", type=str,
                            help="Index file prefix.")
    parser.add_argument("-s", "--start-date", dest="start_date",
                            default=0, type=int,
                            help="The start date for this ne-art index.")
    parser.add_argument("-e", "--end-date", dest="end_date",
                            default=180, type=int,
                            help="The end date for this ne-art index.")

    args = parser.parse_args()
    output_filename = f"{args.output_fileprefix}_{args.start_date}_{args.end_date}.index"

    files = list(glob.glob(args.index_filename))
    ne_dict = defaultdict(list)
    final_ne_dict = defaultdict(list)
    ne_idf_dict = {}
    pair_dict = defaultdict(list) # to compute exact total pair number without duplicate
    total_valid_art_count = 0

    for file in files:
        cur_art_count = 0
        with open(file, "r") as fh:
            for line in fh:
                print(f"loading {file}    {cur_art_count}   {line} ...")

                try:
                    cur_art = (file, cur_art_count)
                    cur_content = line.strip().split("\t")
                    vec = cur_content[-1]
                    vec = ast.literal_eval(vec)

                    for ne_set in vec:
                        ne = ne_set[0]
                        ne_dict[ne].append(cur_art)

                    total_valid_art_count += 1
                except Exception:
                    print(traceback.format_exc())

                cur_art_count += 1

    # compute idf
    for k,v in ne_dict.items():
        ne_idf_dict[k] = np.log(total_valid_art_count/(1 + len(v)))

    # filter according to tf-idf top_k ne ranking per article
    for file in files:
        cur_art_count = -1 # to make sure cur_art_count start from 0
        with open(file, "r") as fh:
            for line in fh:
                print(f"filtering {file}    {cur_art_count}   {line} ...")
                cur_art_count += 1

                try:
                    cur_content = line.strip().split("\t")
                    dat = int(cur_content[-2])
                    if dat < args.start_date or dat > args.end_date:
                        continue
                    vec = cur_content[-1]
                    vec = ast.literal_eval(vec)

                    cur_art = (dat, file, cur_art_count)

                    if len(vec) <= top_k:
                        for ne_set in vec:
                            ne = ne_set[0]
                            final_ne_dict[ne].append(cur_art)

                    else:
                        cur_total_ne_num = sum([ne_set[1] for ne_set in vec])
                        cur_tf_idf_dict = {}

                        for ne_set in vec:
                            ne = ne_set[0]
                            ne_num = ne_set[1]
                            cur_tf_idf_dict[ne] = ne_num/cur_total_ne_num * ne_idf_dict[ne]

                        cur_tf_idf_list = list(cur_tf_idf_dict.values())
                        cur_tf_idf_list.sort(reverse=True)

                        for ne_set in vec:
                            ne = ne_set[0]
                            if cur_tf_idf_dict[ne] >= cur_tf_idf_list[top_k - 1]:
                                final_ne_dict[ne].append(cur_art)

                except Exception:
                    print(traceback.format_exc())


    #save ne_art index
    with open(output_filename, "w") as fh:
        saving_art_count = 0
        saving_pair_count = 0
        for k,v in final_ne_dict.items():
            fh.writelines([str(k), " ", str(len(v))])
            fh.write('\n')
            fh.writelines(str(v))
            fh.write('\n')
            saving_art_count += 1

            # cur_saving_pair_count = 0
            # for art1 in v:
            #     for art2 in v:
            #         lang1 = art1[0].split("/")[-1][:2]
            #         lang2 = art2[0].split("/")[-1][:2]
            #         lang1_idx = lang_list.index(lang1)
            #         lang2_idx = lang_list.index(lang2)
            #         encode_lang_num = lang1_idx * 10 + lang2_idx
            #         encode_pair_key = str(art1[1]) + "_" + str(art2[1])
            #
            #         if (lang1_idx == lang2_idx and art1[1] == art2[1]) or (encode_lang_num in pair_dict[encode_pair_key]):
            #             continue
            #
            #         cur_saving_pair_count += 1
            #         pair_dict[encode_pair_key].append(encode_lang_num) # indicate this pair already exists
            #
            #         print("len(v)", len(v), "cur_saving_pair_count:", cur_saving_pair_count)
            # saving_pair_count += cur_saving_pair_count
            saving_pair_count += len(v) * (len(v)-1)
            print(f"finished saving {saving_art_count} lines....")

        print("completed...........")
        print(f"{saving_pair_count} links in total....")







