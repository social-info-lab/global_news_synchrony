import sys
import os
import io
import json
from string import digits
import gzip

import numpy as np
import nltk
import spacy
from nltk.corpus import stopwords
from tqdm import tqdm
import pathlib
import datetime

def checkNone(it):
    return [] if it == None else it


nltk.download('stopwords')
from nltk.tokenize import word_tokenize

remove_digits = str.maketrans('', '', digits)
ALLOWED_ENTITY_TYPES = {'LAW', 'PRODUCT', 'LOC', 'LANGUAGE', 'NORP', 'WORK_OF_ART', 'GPE', 'EVENT', 'PERSON', 'ORG',
                        'FAC'}
lang_list = ['ar', 'de', 'en', 'es', 'fr', 'it', 'ko', 'pl', 'pt', 'ru', 'tr', 'zh']
# load traffic counts

# merge with page names
print(f"Start Merging page names at {datetime.datetime.now():%Y-%m-%d_%H:%M:%S}")
if not os.path.exists("merged_wiki_data.json"):
    rows = {}
    count = 0
    for line in io.open("wiki_2020_traffic_counts.json", mode="r", encoding="utf-8"):
        count += 1
        row = json.loads(line)
        row["hits_count"] = 0
        if not rows.get(row["title"]):
            rows[row["title"]] = []
        rows[row["title"]].append(row)
        if count % 10000 == 0:
            print(count)

    page_names = []
    outdata = io.open("merged_wiki_data.json", mode="w", encoding="utf-8")
    count = 0
    hits_count = 0
    for line in io.open("wiki_page_names_by_language.json", mode="r", encoding="utf-8"):
        count += 1
        row = json.loads(line)
        row["counts"] = {}
        for language_name, page_name in row["languages"].items():
            if rows.get(page_name.replace(" ", "_")):
                for i, count_row in enumerate(rows.get(page_name.replace(" ", "_"))):
                    if language_name.replace("wiki", "") == count_row.get("wiki_cleaned"):
                        rows[page_name.replace(" ", "_")][i]['hits_count'] += 1
                        row["counts"][language_name] = count_row
                        hits_count += 1
        gg = outdata.write(json.dumps(row, ensure_ascii=False) + "\n")
        if count % 1000 == 0:
            print(count)
            print(hits_count)
print(f"Finish Merging page names at {datetime.datetime.now():%Y-%m-%d_%H:%M:%S}")

# load concepts
print(f"Start Loading Concepts at {datetime.datetime.now():%Y-%m-%d_%H:%M:%S}")
wiki_concepts = {}
for merge_line in io.open("merged_wiki_data.json", mode="r", encoding="utf-8"):
    row = json.loads(merge_line)
    if row["counts"] and len(row["languages"]) > 1:
        for wiki, term in row["languages"].items():
            if not wiki_concepts.get(term):
                wiki_concepts[term] = []
            new_row = {"term_id": row["term_id"], "term": term, "wiki": wiki}
            for countwiki, count_data in row["counts"].items():
                if countwiki == wiki:
                    new_row["views"] = count_data["views"]
            wiki_concepts[term].append(new_row)
print(f"Finish Loading Concepts at {datetime.datetime.now():%Y-%m-%d_%H:%M:%S}")

dataset = sys.argv[1] # can be choose from 'test' and 'train'

for lang in lang_list:
    if os.path.exists(f"/home/xichen/mediacloud/semeval_baseline/semeval_8_2021_ia_data/ner/{dataset}/{lang}/"):
        print(lang)
        run_counts = []
        match_counts = []
        concept_sizes = []
        pathlib.Path(f"/home/xichen/mediacloud/semeval_baseline/semeval_8_2021_ia_data/wikilinked/{dataset}/{lang}/").mkdir(parents=True,
                                                              exist_ok=False)  # Error if there is already data there.
        for filename in tqdm(os.listdir(f"/home/xichen/mediacloud/semeval_baseline/semeval_8_2021_ia_data/ner/{dataset}/{lang}")):
            # print(filename)
            run_count = 0
            match_count = 0
            if ".gz" in filename:
                fh = gzip.open(f"/home/xichen/mediacloud/semeval_baseline/semeval_8_2021_ia_data/ner/{dataset}/{lang}/{filename}", mode="rt", encoding="utf-8")
                outfile = gzip.open(f"/home/xichen/mediacloud/semeval_baseline/semeval_8_2021_ia_data/wikilinked/{dataset}/{lang}/{filename}", mode="wt", encoding="utf-8")
            else:
                fh = io.open(f"/home/xichen/mediacloud/semeval_baseline/semeval_8_2021_ia_data/ner/{dataset}/{lang}/{filename}", mode="r", encoding="utf-8")
                outfile = gzip.open(f"/home/xichen/mediacloud/semeval_baseline/semeval_8_2021_ia_data/wikilinked/{dataset}/{lang}/{filename}.gz", mode="wt", encoding="utf-8")
            for lineno, article_line in enumerate(fh):
                try:
                    article_entities = json.loads(article_line)
                except:
                    sys.stderr.write(f"Error processing line {lineno} of {filename}.\n")
                    sys.stderr.write(article_line)
                    sys.stderr.write("\n")
                    continue
                article_entities["wiki_concepts"] = []
                for entity in checkNone(article_entities.get('spacy', [])):
                    run_count += 1
                    if wiki_concepts.get(entity['text']):
                        wiki_concepts.get(entity['text']).sort(key=lambda x: x.get("views", 0))
                        article_entities["wiki_concepts"].append(wiki_concepts.get(entity['text'])[-1])
                        match_count += 1
                for entity in checkNone(article_entities.get('polyglot', [])):
                    run_count += 1
                    for each_entity in entity['text']:
                        if wiki_concepts.get(each_entity):
                            wiki_concepts.get(each_entity).sort(key=lambda x: x.get("views", 0))
                            article_entities["wiki_concepts"].append(wiki_concepts.get(each_entity)[-1])
                            match_count += 1
                print(f"processed {run_count} articles for {lang}")
                concept_size = []
                for concept in checkNone(article_entities.get('wiki_concepts', [])):
                    concept_size.append(len(wiki_concepts.get(concept['term'], [])))
                if concept_size:
                    concept_sizes.append(np.mean(concept_size))
                else:
                    concept_sizes.append(0.0)
                del article_entities["text"]  # Remove full text of article.
                gz = outfile.write(json.dumps(article_entities) + "\n")
            outfile.close()
            run_counts.append(run_count)
            match_counts.append(match_count)
