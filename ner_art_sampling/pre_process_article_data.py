import pandas as pd
import json
import collections
import os
import glob
from argparse import ArgumentParser

# articles = pd.read_json("stories_0.json")

lang_list = ['ar', 'de', 'en', 'es', 'fr', 'it', 'pl', 'ru', 'urk', 'zh']

if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument("-s", "--source-dir", dest="source_dir",
                        default="/home/alizeynali/MediaAPI/UK_RU", type=str,
                        help="source directory")
    parser.add_argument("-t", "--target-dir", dest="target_dir",
                        default="/home/xichen/mediacloud/ner_art_sampling/UK_RU", type=str,
                        help="target directory")

    args = parser.parse_args()

    # source_dir = "/home/alizeynali/MediaAPI/UK_RU"
    # target_dir = "/home/xichen/mediacloud/ner_art_sampling/UK_RU"
    # source_dir = "/home/alizeynali/MediaAPI/UK_RU_0"
    # target_dir = "/home/xichen/mediacloud/ner_art_sampling/UK_RU_0"
    # source_dir = "/home/alizeynali/MediaAPI/UK_RU_May_2020"
    # target_dir = "/home/xichen/mediacloud/ner_art_sampling/UK_RU_May_2020"

    source_dir, target_dir = args.source_dir, args.target_dir

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for lang in lang_list:
        lang_dir = f"{source_dir}/{lang}"
        splitted_by_date_dir = f"{target_dir}/splitted_by_date"
        splitted_by_date_lang_dir =  f"{splitted_by_date_dir}/{lang}"
        if not os.path.exists(splitted_by_date_dir):
            os.makedirs(splitted_by_date_dir)
        if not os.path.exists(splitted_by_date_lang_dir):
            os.makedirs(splitted_by_date_lang_dir)

        dated_art = collections.defaultdict(list)
        splitted_count = 0
        processed_count = 0

        files = glob.glob(lang_dir + "/*.json")
        for f in files:
            with open(f, "r") as fh:
                art_sets = json.load(fh)
                # for lineno, line in enumerate(fh):
                #     art_sets = json.loads(line)
                for art in art_sets:
                    date = art['collect_date'][0:10]
                    dated_art[date].append(json.dumps(art))
                    splitted_count += 1
                    print(f"{lang}: splitted {splitted_count} articles ...")

        for k, v in dated_art.items():
            with open(f"{splitted_by_date_lang_dir}/{k}.json", "a") as fh:
                for art in v:
                    fh.write(art)
                    fh.write("\n")
                    processed_count += 1
                    print(f"{lang}: processed {processed_count} articles ...")