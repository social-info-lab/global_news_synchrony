import glob
import csv
import json
import pandas as pd
import re
import jieba
import sklearn.metrics
import scipy.stats
import datetime
import socket
import ast
import collections
import gc
import random
import gzip
import sys
import numpy as np
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import classification_report
from sklearn.metrics import auc, matthews_corrcoef, f1_score, precision_score, accuracy_score, recall_score
from sklearn.metrics import confusion_matrix
from scipy.stats import pearsonr
from matplotlib import pyplot
from matplotlib import pylab


News = collections.namedtuple("News", "file, lineno, url, vec")

# unify the labels of each pair and enlarge the data size to 10 times
def unify_data_labels(data,column_names):
    new_data = pd.DataFrame(columns=column_names)
    for i in range(len(data)):
        a = data.loc[i]
        d = pd.DataFrame(a).T
        new_data = new_data.append([d] * 10)
    new_data["modified_overall"] = new_data.apply(lambda x : unify_label_with_prob(x["Overall"]), axis=1)

    return new_data

def unify_label_with_prob(label_score):
    if not isinstance(label_score, float):
        label_score = float(label_score)
    int_label_score = int(label_score)
    res = label_score - int_label_score

    return int_label_score + np.random.binomial(1, res, 1)[0]
def read_index_data_for_match2(index_filename):
    data = []
    print(datetime.datetime.now(), f"Reading data from {index_filename}")
    with open(index_filename, "r") as fh:
        n_lines = 0
        print("Loaded...", end=" ")
        for line in fh:
            file, lineno, reladate, url, vec = line.strip().split("\t")
            # we've heavily oversampled this day, so now we can undersample it
            # please move this step to create_index once index versioning and tracking is implemented (we need similar tracking for the remaining two scripts as well)
            # it's easy to forget it, especially on a local machine with alternative setup and small data samples
            # all article-specific filtering should happen at the index level anyway

            # if "2020-01-01" in file:
            # 	if socket.gethostname()!="pms-mm.local":
            # 		continue

            vec = tuple(ast.literal_eval(vec))
            # tuples have lower footprint than lists, lists have lower than sets
            # https://stackoverflow.com/questions/46664007/why-do-tuples-take-less-space-in-memory-than-lists/46664277
            # https://stackoverflow.com/questions/39914266/memory-consumption-of-a-list-and-set-in-python

            # # we don't consider the repeat times of words in the same document here
            # vec = tuple(k for k, v in vec)
            tup = News(file, lineno, url, vec)
            data.append( tup )
            n_lines += 1
            if n_lines%500000 == 0:
                print(n_lines,"lines", end=" ", flush=True)
                gc.collect()
    print(datetime.datetime.now(), "Loaded", len(data), "news from",index_filename)
    # we want to get different pairs every time, some pairs may randomly repeat,
    # but chances of this are low and we can deal with them later
    # before sending them to annotators
    random.seed()
    random.shuffle(data)
    print(datetime.datetime.now(), "Shuffled the index", index_filename)

    gc.collect()
    return data

def load_article(file,lineno):
    try:
        lineno=int(lineno)
    except ValueError:
        print("DEBUG", lineno, "Note: known issue, needs debugging.")
        # probably some issue about saving/loading namedtuples
        sys.exit()
    # symbolic link doesn't look good with current inter-lang data storage format since the current wiki data is in scott's directory while the offsets are in my directory. Even use symbolic links it will always change the code since we use the json files and offsets at different time.
    with open(file.replace(".json", ".offsets").replace('-REM','').replace(".gz", "").replace("home/scott/wikilinked","home/xichen/mediacloud/ner_art_sampling/wikilinked"), "r") as fh:
    # with open(file.replace(".json", ".offsets").replace(".gz", ""), "r") as fh:
        offsets = [int(x) for x in fh]
    if ".gz" not in file:
        with open(file,"r") as fh:
            fh.seek(offsets[lineno])
            line=fh.readline()
    else:
        with gzip.open(file,"rt") as fh:
            fh.seek(offsets[lineno])
            line=fh.readline()
    return json.loads(line)

def match_url_to_ner_and_wiki(url1_list, url2_list, index_dict):
    # change list to dict to save search time
    url_dict = {}
    url_ner_dict = collections.defaultdict(dict)
    url_wiki_dict = collections.defaultdict(dict)
    for url in url1_list:
        url_dict[url.strip()] = 1
    for url in url2_list:
        url_dict[url.strip()] = 1

    for lang in index_dict.keys():
        count = 0
        for art in index_dict[lang]:
            count += 1
            print(lang, "    ", count, 'searched..      ')
            if art.url in url_dict:
                if len(lang) == 2:
                    url_ner_dict[art.url.strip()]['ne_list'] = art.vec
                    url_ner_dict[art.url.strip()]["text"] = load_article(art.file, art.lineno)["story_text"]
                else:
                    url_wiki_dict[art.url.strip()]['ne_list'] = art.vec
                    url_wiki_dict[art.url.strip()]['text'] = load_article(art.file.replace("home/scott/wikilinked", "home/scott/ner").replace('-REM','').replace(".gz",""), art.lineno)["story_text"]
    print(len(url_ner_dict.keys()), len(url_wiki_dict.keys()) )
    # print(url1_ner_dict)

    return url_ner_dict, url_wiki_dict

def unify_ner(inter_type, lang, art):
    vec= []
    if inter_type == "intra-lang":
        if "spacy" in art and art["spacy"] != None:
            for ent in art["spacy"]:
                text = ent["text"]
                if len(text) < 2 and lang not in ["zh", "ko"]:
                    continue
                vec.append(text)
        if "polyglot" in art and art["polyglot"] != None:
            for ent in art["polyglot"]:
                # the implemention used by xi chen
                for word in ent["text"]:
                    if len(word) < 2 and lang not in ["zh", "ko"]:
                        continue
                    vec.append(word)
    elif inter_type == "inter-lang":
        if "wiki_concepts" in art and art["wiki_concepts"] != None:
            for ent in art["wiki_concepts"]:
                if len(ent['term']) < 2 and lang not in ["zh", "ko"]:
                    continue
                vec.append(ent['term'])
    return vec

def ner_jaccard_sim(input_type, lang1, lang2, art1, art2):
    ne_list1 = art1['ne_list']
    ne_list2 = art2['ne_list']

    intersect = {}
    union = {}

    for each in ne_list1:
        union[each] = 1
    for each in ne_list2:
        union[each] = 1

    for each1 in ne_list1:
        for each2 in ne_list2:
            if each1 == each2:
                intersect[each1] = 1
                break

    ner_jaccard_sim = len(intersect.keys())/len(union.keys())

    return ner_jaccard_sim

def text_jaccard_sim(lang1, lang2, text1, text2):
    text_list1 = text2tokens(text1, lang1)
    text_list2 = text2tokens(text2, lang2)

    intersect = {}
    union = {}

    for each in text_list1:
        union[each] = 1
    for each in text_list2:
        union[each] = 1

    for each1 in text_list1:
        for each2 in text_list2:
            if each1 == each2:
                intersect[each1] = 1
                break

    text_jaccard_sim = len(intersect.keys())/len(union.keys())

    return text_jaccard_sim

def text2tokens(txt, lang):
    if not isinstance(txt, str):
        txt = str(txt)

    txt=re.sub("[^\w ]","",txt) #only letters and spaces
    if lang=="zh":
        words = " ".join(jieba.cut(txt)).split()
    else:
        words = txt.split(" ")
    for i in range(len(words)):
        # here we choose the first 5 decimal places of the hash value to fit the int range of cython
        words[i] = int(hash(words[i])/(10**14))
    words.sort()

    return tuple(words)

def compute_pair_ner_jaccard_sim(lang1, lang2, url1, url2, ner_dict, wiki_dict):
    try:
        if lang1 == lang2:
            return ner_jaccard_sim("intra-lang", lang1, lang2, ner_dict[url1], ner_dict[url2])
        else:
            return ner_jaccard_sim("inter-lang", lang1, lang2, wiki_dict[url1], wiki_dict[url2])
    except:
        print("url1 not in the dict:", url1)
        print("url2 not in the dict:", url2)
        return 0

def compute_pair_text_jaccard_sim(lang1, lang2, url1, url2, ner_dict, wiki_dict):
    try:
        return text_jaccard_sim(lang1, lang2, ner_dict[url1]['text'], ner_dict[url2]['text'])
    except:
        return 0

def text_len_diff(lang1, lang2, url1, url2, ner_dict, wiki_dict):
    try:
        text_list1 = text2tokens(ner_dict[url1]['text'], lang1)
        text_list2 = text2tokens(ner_dict[url2]['text'], lang2)

        return abs(len(text_list1) - len(text_list2))
    except:
        return 10000


# loading sample pairs
train_data = pd.read_csv("semeval-2022_task8_train-data_batch.csv")
test_data = pd.read_csv("unreleased_ia_filtered_with_score_reversed_filtered.csv")

# loading name entities
train_directory = 'semeval_8_2021_ia_data/train'
test_directory = 'semeval_8_2021_ia_data/test'

index_directory = "/home/xichen/mediacloud/ner_art_sampling/indexes"

selected_data_lang = []
selected_index_dict = {}

for each in train_data['url1_lang']:
    if each not in selected_data_lang:
        selected_data_lang.append(each)
for each in train_data['url2_lang']:
    if each not in selected_data_lang:
        selected_data_lang.append(each)
# for test
# selected_data_lang = ['pl']
# selected_data_lang = []
for each_lang in selected_data_lang:
    selected_index_dict[each_lang] = read_index_data_for_match2(index_directory + f'/{each_lang}-v2.index')
    selected_index_dict[each_lang+'-wiki'] = read_index_data_for_match2(index_directory + f'/{each_lang}-wiki-v2.index')

train_ner_dict, train_wiki_dict = match_url_to_ner_and_wiki(train_data['link1'], train_data['link2'], selected_index_dict)
test_ner_dict, test_wiki_dict = match_url_to_ner_and_wiki(test_data['link1'], test_data['link2'], selected_index_dict)

print("train data: computing ne_jaccard_sim")
train_data['ne_jaccard_sim'] = train_data.apply(lambda x: compute_pair_ner_jaccard_sim(x['url1_lang'], x['url2_lang'], x['link1'].strip(), x['link2'].strip(), train_ner_dict, train_wiki_dict), axis=1)
print("train data: computing text_jaccard_sim")
train_data['text_jaccard_sim'] = train_data.apply(lambda x: compute_pair_text_jaccard_sim(x['url1_lang'], x['url2_lang'], x['link1'].strip(), x['link2'].strip(), train_ner_dict, train_wiki_dict), axis=1)
print("train data: computing text_len_dff")
train_data['text_len_diff'] = train_data.apply(lambda x: text_len_diff(x['url1_lang'], x['url2_lang'], x['link1'].strip(), x['link2'].strip(), train_ner_dict, train_wiki_dict), axis=1)

print("test data: computing ne_jaccard_sim")
test_data['ne_jaccard_sim'] = test_data.apply(lambda x: compute_pair_ner_jaccard_sim(x['url1_lang'], x['url2_lang'], x['link1'].strip(), x['link2'].strip(), test_ner_dict, test_wiki_dict), axis=1)
print("test data: computing text_jaccard_sim")
test_data['text_jaccard_sim'] = test_data.apply(lambda x: compute_pair_text_jaccard_sim(x['url1_lang'], x['url2_lang'], x['link1'].strip(), x['link2'].strip(), test_ner_dict, test_wiki_dict), axis=1)
print("test data: computing text_len_dff")
test_data['text_len_diff'] = test_data.apply(lambda x: text_len_diff(x['url1_lang'], x['url2_lang'], x['link1'].strip(), x['link2'].strip(), test_ner_dict, test_wiki_dict), axis=1)

# train_data['modified_overall'] = train_data.apply(lambda x: round(x["Overall"]), axis=1)
'''unifying labels in train data according to a probablistic manner'''
print("raw_train_data_size:", len(train_data))
train_data = unify_data_labels(train_data, ['url1_lang', 'url2_lang', 'pair_id', 'link1', 'link2', 'ia_link1', 'ia_link2', 'Geography', 'Entities', 'Time', 'Narrative', 'Overall', 'Style', 'Tone','ne_jaccard_sim', 'text_jaccard_sim', 'text_len_diff'])
new_train_data = train_data.loc[train_data['ne_jaccard_sim'] != 0]
# print(new_train_data.columns.values)
# new_train_data = new_train_data.drop(columns=["Unnamed: 0"])
train_data = new_train_data
print("fiterled_train_data_size:", len(train_data))

train_data.to_csv("train_data_for_baseline_with_index_data.csv")
test_data.to_csv("test_data_for_baseline_with_index_data.csv")



x_train1 = train_data[['ne_jaccard_sim']]
x_test1 = test_data[['ne_jaccard_sim']]
x_train2 = train_data[['ne_jaccard_sim','text_jaccard_sim']]
x_test2 = test_data[['ne_jaccard_sim','text_jaccard_sim']]
x_train3 = train_data[['ne_jaccard_sim','text_jaccard_sim','text_len_diff']]
x_test3 = test_data[['ne_jaccard_sim','text_jaccard_sim','text_len_diff']]

y_train = train_data['modified_overall']
y_test = test_data['Overall'] # the labels are float since we only need to compute pearson correlation now

std = StandardScaler()
x_train1 = std.fit_transform(x_train1)
x_test1 = std.transform(x_test1)
x_train2 = std.fit_transform(x_train2)
x_test2 = std.transform(x_test2)
x_train3 = std.fit_transform(x_train3)
x_test3 = std.transform(x_test3)

# x_train1
model_name = "SVC-linear"
print(model_name + '...')
model = SVC(kernel='linear',gamma='auto', class_weight='balanced')
model.fit(x_train1, y_train)
y_predict1 = model.predict(x_test1)

# regression coefficient
print(model.coef_)  # [[1.12796779  0.28741414  0.66944043 ...]]

# compute metrics
result = {}
pearson_cor = pearsonr(y_test, y_predict1)
print("pearson_cor:", pearson_cor)

output = pd.DataFrame()
output['pair_id'] = test_data['pair_id']
output['Overall'] = y_predict1.T
output.to_csv("baseline_prediction_with_index_data_train1.csv")

# x_train2
model_name = "SVC-linear"
print(model_name + '...')
model = SVC(kernel='linear',gamma='auto', class_weight='balanced')
model.fit(x_train2, y_train)
y_predict2 = model.predict(x_test2)

# regression coefficient
print(model.coef_)  # [[1.12796779  0.28741414  0.66944043 ...]]

# compute metrics
result = {}
pearson_cor = pearsonr(y_test, y_predict2)
print("pearson_cor:", pearson_cor)

output = pd.DataFrame()
output['pair_id'] = test_data['pair_id']
output['Overall'] = y_predict2.T
output.to_csv("baseline_prediction_with_index_data_train2.csv")

# x_train3
model_name = "SVC-linear"
print(model_name + '...')
model = SVC(kernel='linear',gamma='auto', class_weight='balanced')
model.fit(x_train3, y_train)
y_predict3 = model.predict(x_test3)

# regression coefficient
print(model.coef_)  # [[1.12796779  0.28741414  0.66944043 ...]]

# compute metrics
result = {}
pearson_cor = pearsonr(y_test, y_predict3)
print("pearson_cor:", pearson_cor)

output = pd.DataFrame()
output['pair_id'] = test_data['pair_id']
output['Overall'] = y_predict3.T
output.to_csv("baseline_prediction_with_index_data_train3.csv")
