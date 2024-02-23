from functools import partial
import os
#import multiprocessing
import concurrent.futures
import sys
import json
import spacy
import collections
#import gcld3
import time
import csv
import datetime

def parse_spacy(text, model):
    # lang = detector.FindLanguage(text=text).language
    return [
        {
            "text": ent.text,
            "start_char": ent.start_char,
            "end_char": ent.end_char,
            "label": ent.label_
        } for ent in model(text).ents
    ]


def process_article(raw_article, model):
    text = ''.join([ch for ch in raw_article["story_text"] if ch.isprintable()])
    if text == None:
        return json.dumps(raw_article)
    # lang = raw_article["language"]  # detector.FindLanguage(text=text).language
    row = raw_article
    try:
        row["spacy"] = parse_spacy(text, model)
    except:
        row["spacy"] = None

    publish_date = raw_article['publish_date']
    if publish_date == None:
        publish_date = "NAN"
    else:
        publish_date = publish_date[:10]

    return publish_date, json.dumps(row)


def process_file(filename, model):
    print(filename)
    story_id = filename.split('/')[-1].split('.')[0]
    articles = []
    with open(filename, "r") as fh:
        for line in fh.readlines():
            cur_art = json.loads(line)
            articles.append(cur_art)
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
    # input_directory = "UK_RU/splitted_by_date"
    # output_directory = "UK_RU/splitted_by_date/ner"
    input_directory = sys.argv[1]
    output_directory = sys.argv[2]
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    lang_list = ["en", "de", "es", "fr", "it", "pl", "zh", "ru"]


    # Assume all articles are in the correct language
    # will greatly reduce memory as we can load only ONE spacy model
    nlp = {
        "en": "en_core_web_sm",  # spacy.load("en_core_web_sm"),
        "de": "de_core_news_sm",
        "es": "es_core_news_sm",
        "pl": "pl_core_news_sm",
        "pt": "pt_core_news_sm",
        "zh": "zh_core_web_sm",
        "fr": "fr_core_news_sm",
        "ru": "ru_core_news_sm",
        "it": "it_core_news_lg",
    }

    for lang in lang_list:
        cur_output_directory = output_directory + "/" + lang
        if not os.path.exists(cur_output_directory):
            os.makedirs(cur_output_directory)

        files = glob.glob(input_directory + f"/{lang}/*.json")
        print()
        print(len(files))

        # how long it takes?
        print(f"loading model for {lang} start from {datetime.datetime.now():%Y-%m-%d_%H:%M:%S}", flush=True)
        model = spacy.load(nlp[lang], disable=["tok2vec", "tagger", "parser", "lemmatizer", "attribute_ruler"])
        print(f"loading model for {lang} finish at {datetime.datetime.now():%Y-%m-%d_%H:%M:%S}", flush=True)

        pfile = partial(process_file, model=model)
        tofiles = partial(append_to_files, output_dir=cur_output_directory)
        logerr = partial(log_error, output_dir=cur_output_directory)
        # with multiprocessing.Pool(processes=1,maxtasksperchild=100) as pool: #2*os.cpu_count()
        with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
            for f in files:
                with open(f, "r") as fh:
                    future = executor.submit(pfile, f)
                    future.add_done_callback(tofiles)

        print("DONE")
