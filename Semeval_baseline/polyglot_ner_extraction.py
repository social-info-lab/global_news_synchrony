import traceback
from functools import partial
import os
# import multiprocessing
import concurrent.futures
import sys
import json
# import spacy
import polyglot
from polyglot.text import Text, Word
import collections
import csv
# import gcld3
import time
from polyglot.downloader import downloader
downloader.download("embeddings2.ar")
downloader.download("embeddings2.tr")
downloader.download("embeddings2.ko")
# downloader.download("embeddings2.un")
downloader.download("ner2.ar")
downloader.download("ner2.tr")
downloader.download("ner2.ko")
# downloader.download("ner2.un")

def parse_polyglot_entity(entity):
    return {
        "text": json.loads(json.dumps(entity)),
        "type": entity.tag
    }


def parse_polyglot(text):
    try:
        return [parse_polyglot_entity(e) for e in Text(text).entities]
    except:
        traceback.print_exc()
        return []


def process_article(raw_article, model=None):
    text = ''.join([ch for ch in raw_article["text"] if ch.isprintable()])
    if text == None:
        return json.dumps(raw_article)
    row = raw_article
    try:
        row["polyglot"] = parse_polyglot(text)
    except:
        row["polyglot"] = None

    publish_date = raw_article['publish_date']
    if publish_date == None:
        publish_date = "NAN"
    else:
        publish_date = publish_date[:10]

    return publish_date, json.dumps(row)


def process_file(filename, model):
    print(filename)
    story_id = filename.split('/')[-1].split('.')[0]

    with open(filename, "r") as fh:
        articles = json.load(fh)
    if isinstance(articles, dict):
        articles = [articles]
    output = collections.defaultdict(list)
    for art in articles:
        art['story_id'] = story_id
        date, dat = process_article(art, model)
        output[date].append(dat)
    return output


PROCESSED_FILES = 0


def append_to_files(future, output_dir):
    global PROCESSED_FILES
    PROCESSED_FILES += 1
    data = future.result()
    for k, v in data.items():
        with open(f"{output_dir}/{k}.json", "a") as fh:
            for art in v:
                fh.write(art)
                fh.write("\n")
    print(f"{PROCESSED_FILES} completed")


def log_error(e, output_dir):
    with open(output_dir + "/ERR.txt", "a") as fh:
        print(e)
        fh.write(str(e))


import glob

if __name__ == "__main__":
    input_directory = sys.argv[1]
    output_directory = sys.argv[2]
    csv_file = sys.argv[3]
    # input_directory = "semeval_8_2021_ia_data/train"
    # output_directory = "semeval_8_2021_ia_data/ner/train"
    # csv_file = "semeval-2022_task8_train-data_batch.csv"

    # input_directory = "semeval_8_2021_ia_data/test"
    # output_directory = "semeval_8_2021_ia_data/ner/test"
    # csv_file = "semeval-2022_task8_eval_data_202201.csv"


    # loading language from the csv file
    lang_id_dict = collections.defaultdict(list)
    id_lang_dict = collections.defaultdict(str)
    csv_reader = csv.reader(open(csv_file))
    for line in csv_reader:
        story_id1, story_id2 = line[2].split('_')
        lang_id_dict[line[0]].append(story_id1)
        lang_id_dict[line[1]].append(story_id2)

        id_lang_dict[story_id1] = line[0]
        id_lang_dict[story_id2] = line[1]

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    polyglot_lang = ["ar","tr","ko"]

    files = glob.glob(input_directory + "/*/*.json")
    print()
    print(len(files))

    count = 0
    for lang in polyglot_lang:
        cur_output_directory = output_directory + "/" + lang
        if not os.path.exists(cur_output_directory):
            os.makedirs(cur_output_directory)

        pfile = partial(process_file, model=None)
        tofiles = partial(append_to_files, output_dir=cur_output_directory)
        logerr = partial(log_error, output_dir=cur_output_directory)
        # with multiprocessing.Pool(processes=1,maxtasksperchild=100) as pool: #2*os.cpu_count()
        with concurrent.futures.ProcessPoolExecutor(max_workers=25) as executor:
            for f in files:
                cur_story_id = f.split('/')[-1].split('.')[0]
                with open(f, "r") as fh:
                    articles = json.load(fh)
                    if id_lang_dict[cur_story_id] == lang:
                        future = executor.submit(pfile, f)
                        future.add_done_callback(tofiles)
    print("DONE")