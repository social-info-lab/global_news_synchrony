""""""
'''sbatch -o script_output/visualize_matched_inference/visualize_matched_inference_gm.txt visualize_matched_inference_script.sh network_pairs/matched_prediction/integrated/integrated_matched_prediction_0_180.csv gm '''

# option list
# gm: Visualize the network inference on the map, node size is the intra-lang similarity, while edge width is the inter-lang similarity.
# da: Analyze similarity in terms of country and language with different democracy index.
# ta: Analyze similarity in terms of country and language over time.

import collections
from itertools import combinations
from argparse import ArgumentParser
from datetime import datetime
import json
from collections import defaultdict
import numpy as np
import scipy.stats as st
import pandas as pd
import matplotlib.pyplot as plt
import math
import operator
import os
import geopy.distance
import heapq
import seaborn as sns
import random
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.iolib.summary2 import summary_col
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import copy
import gzip
import sys
import traceback
import igraph
import csv
import ssl
import irrCAC
import jsonlines

import shap


import urllib.request
import urllib.parse

import socket
import ast
import gc

from utils import News

from pyecharts import options as opts
from pyecharts.charts import Geo
from pyecharts.globals import ChartType, SymbolType, GeoType
from pyecharts.charts import Map
import glob
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from scipy.spatial import distance
from scipy.stats import entropy
from statsmodels.iolib.summary import Summary

from utils import unify_url, trans_node_sim,trans_edge_sim, trans_dem_idx_sim, democracy_index_class, distance_class, match_to_distance_class, political_group_list, months, month_len, lang_dict, lang_full_name_dict, LANG_FULL_NAME_MAP, LANG_FAMILY
from utils import lang_list



def match_within_political_group(country_1, country_2, political_group_list, political_group_dict):
    matched_within_political_groups = []
    for group in political_group_list:
        if (country_1 in political_group_dict[group]) and (country_2 in political_group_dict[group]):
            matched_within_political_groups.append(group)
    return matched_within_political_groups

def match_across_political_group(country_1, country_2, political_group_list, political_group_dict):
    if country_1 == country_2:
        return []

    matched_across_political_groups = {}
    matched_political_groups1 = []
    matched_political_groups2 = []

    for group in political_group_list:
        if (country_1 in political_group_dict[group]):
            matched_political_groups1.append(group)
        if (country_2 in political_group_dict[group]):
            matched_political_groups2.append(group)

    for group1 in matched_political_groups1:
        for group2 in matched_political_groups2:
            if group1 != group2:
                if group1 > group2:
                    combo = group1 + "-" + group2
                else:
                    combo = group2 + "-" + group1
                matched_across_political_groups[combo] = 1

    return matched_across_political_groups.keys()


def find_unitary_type(country, country_unitary_dict):
    for unitary_type in country_unitary_dict:
        if country in country_unitary_dict[unitary_type]:
            return unitary_type
    return

def dem_class_count(dem_country_count_list, pair_type):
    dem_class_count = 0
    if pair_type == "intra_country":
        for count in dem_country_count_list:
            dem_class_count += count * (count-1)
    elif pair_type == "inter_country":
        for i in range(len(dem_country_count_list)):
            for j in range(len(dem_country_count_list)):
                if i !=j:
                    dem_class_count += dem_country_count_list[i] * dem_country_count_list[j]
        dem_class_count = dem_class_count/2 # i and j are computed twice
    return dem_class_count

def next_possible_feature(X_npf, y_npf, current_features, ignore_features=[]):
    '''
    This function will loop through each column that isn't in your feature model and
    calculate the r-squared value if it were the next feature added to your model.
    It will display a dataframe with a sorted r-squared value.
    X_npf = X dataframe
    y_npf = y dataframe
    current_features = list of features that are already in your model
    ignore_features = list of unused features we want to skip over
    '''
    #Create an empty dictionary that will be used to store our results
    function_dict = {'predictor': [], 'r-squared':[]}
    #Iterate through every column in X
    shuffle_cols = list(X_npf.columns)
    random.shuffle(shuffle_cols)
    for col in shuffle_cols:
        #But only create a model if the feature isn't already selected or ignored
        if col not in (current_features+ignore_features):
            #Create a dataframe called function_X with our current features + 1
            selected_X = X_npf[current_features + [col]]
            #Fit a model for our target and our selected columns
            model = sm.OLS(y_npf, sm.add_constant(selected_X)).fit()
            #Predict what  our target would be for our selected columns
            y_preds = model.predict(sm.add_constant(selected_X))
            #Add the column name to our dictionary
            function_dict['predictor'].append(col)
            #Calculate the r-squared value between the target and predicted target
            r2 = np.corrcoef(y_npf, y_preds)[0, 1]**2
            #Add the r-squared value to our dictionary
            function_dict['r-squared'].append(r2)
    # Once it's iterated through every column, turn our dict into a sorted DataFrame
    function_df = pd.DataFrame(function_dict).sort_values(by=['r-squared'], ascending=False)

    return function_df.iloc[0]


def read_storyid_url_index(art_dict, index_filename):
    print(datetime.now(), f"Reading data from {index_filename}")
    with open(index_filename, "r") as fh:
        n_lines = 0
        print("Loaded...", end=" ")
        for line in fh:
            storyid, file, lineno, url  = line.strip().split("\t")

            art_dict[int(storyid)] = (file, lineno, url)

            n_lines += 1
            if n_lines%500000 == 0:
                print(n_lines,"lines", end=" ", flush=True)
                gc.collect()
    print(datetime.now(), "Loaded", n_lines, "news from",index_filename)

    gc.collect()
    return 0


def read_story_wiki_data(index_filename, art_dict, title_dict, file_dict, lineno_dict):
    print(datetime.now(), f"Reading data from {index_filename}")
    with open(index_filename, "r") as fh:
        n_lines = 0
        print("Loaded...", end=" ")
        for line in fh:
            # since we added the title, some titles occupy multiple lines
            try:
                stories_id, title, file, lineno, reladate, url, ne = line.strip().split("\t")

                art_dict[int(stories_id)] = url
                title_dict[int(stories_id)] = title
                file_dict[int(stories_id)] = file
                lineno_dict[int(stories_id)] = lineno
            except:
                pass

            n_lines += 1
            if n_lines%500000 == 0:
                print(n_lines,"lines", end=" ", flush=True)
                gc.collect()
    print(datetime.now(), "Loaded", n_lines, "news from",index_filename)

    gc.collect()
    return 0


def quoteurl(url):
    doubleslash = url.find("//")
    nonhost = url.find("/", doubleslash + 2)
    return url[0:nonhost] + urllib.parse.quote(url[nonhost:])

def checkurl(url):
    # for u in [url, f"https://web.archive.org/web/{url}"]:
    live_status = None
    try:
        url.encode('ascii')
    except:
        url = quoteurl(url)

    # iaurl = f"http://archive.org/wayback/available?url={url}"
    iaurl = f"https://archive.org/wayback/available?url={url}"
    resp = None
    ia_status = None
    try:
        resp = urllib.request.urlopen(iaurl, context=ssl_context)
        resp = json.loads(resp.read())
        # {"url": "example.com", "archived_snapshots": {"closest": {"status": "200", "available": true, "url": "http://web.archive.org/web/20210716023754/https://example.com/", "timestamp": "20210716023754"}}}
        for snap in resp["archived_snapshots"]:
            s = resp["archived_snapshots"][snap]["status"]
            try:
                s = int(s)
            except:
                continue
            if s >= 200 and s < 300:
                return {"outcome": "ia", "live_status": live_status, "ia": resp, "ia_status": ia_status}
    except Exception as e:
        print(e)
    # pass
    return {"outcome": False, "live_status": live_status, "ia": resp, "ia_status": ia_status}


if __name__ == '__main__':
    parser = ArgumentParser()
    # parser.add_argument("-i", "--input-file", dest="input_file",
    #                         default="network_pairs/matched_prediction/integrated/integrated_matched_prediction_0_180.csv", type=str,
    #                         help="Input file.")
    parser.add_argument("-i", "--input-file", dest="input_file",
                            default="network_pairs/matched_prediction/integrated/integrated_matched_prediction_165_180.csv", type=str,
                            help="Input file.")
    parser.add_argument("-o", "--output-dir", dest="output_dir",
                            default="figs/integrated_matched_prediction", type=str,
                            help="output directory.")
    parser.add_argument("-cls", "--cluster-sim", dest="cluster_sim",
                            default=True, action='store_true',
                            help="switch average similarity to cluster-based similarity as per each country.")
    parser.add_argument("-s", "--stat", dest="stat",
                            default=False, action='store_true',
                            help="integrate index statistics for plotting the fraction of filtered-in article/article pairs.")
    parser.add_argument("-opt", "--option", dest="option",
                            default="rm_intra", type=str,
                            help="can be chosen from stat, centrality, save, gm, da, pfa, ta, dta, ga, rm, rm_intra, oslom_rm, oslom_stats, oslom_temp, plt, sim_cmp, cls_eval, none.")
    args = parser.parse_args()
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create a secure context
    ssl_context = ssl.create_default_context()

    # Disable SSL verification (not recommended for production)
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    a = checkurl("https://www.repubblica.it/cronaca/2020/03/08/news/coronavirus_i_decreti_del_governo-250617415/")

    print()



    # load article pairs data into df
    pairs = pd.read_csv(args.input_file)
    pairs["similarity"] = pairs.apply(lambda x: (5-x["similarity"]), axis=1)
    pairs_dict = {key: pairs[key].to_list() for key in pairs.keys()}

    pair_pub_country = pairs.dropna(subset=["pub_country1","pub_country2"])
    pair_about_country = pairs.dropna(subset=["about_country1","about_country2"])

    country_alpha3_to_full_name = {}
    for i in range(len(pairs)):
        country_alpha3_to_full_name[pairs_dict["main_country1"][i]] = pairs_dict["country_full_name1"][i]
        country_alpha3_to_full_name[pairs_dict["main_country2"][i]] = pairs_dict["country_full_name2"][i]

    country_gdp = pd.read_csv("country_info/country_gdp.csv")
    country_gdp = country_gdp[["Country Name", "2020"]]
    country_gdp.rename(columns={"Country Name": 'Country'}, inplace=True)
    country_gdp.rename(columns={"2020": '2020_gdp'}, inplace=True)
    country_gdp_dict = {}
    for country in country_gdp.itertuples():
        country_gdp_dict[country[1]] = country[2]

    country_gdp_per_person = pd.read_csv("country_info/country_gdp_per_person.csv")
    country_gdp_per_person = country_gdp_per_person[["Country Name", "2020"]]
    country_gdp_per_person.rename(columns={"Country Name": 'Country'}, inplace=True)
    country_gdp_per_person.rename(columns={"2020": '2020_gdp_per_person'}, inplace=True)
    country_gdp_per_person_dict = {}
    for country in country_gdp_per_person.itertuples():
        country_gdp_per_person_dict[country[1]] = country[2]

    pairs["country_gdp1"] = pairs.apply(lambda x: country_gdp_dict[x["country_full_name1"]] if (x["country_full_name1"] in country_gdp_dict) else np.nan, axis=1)
    pairs["country_gdp2"] = pairs.apply(lambda x: country_gdp_dict[x["country_full_name2"]] if (x["country_full_name2"] in country_gdp_dict) else np.nan, axis=1)

    pairs["country_gdp1_per_person"] = pairs.apply(lambda x: country_gdp_per_person_dict[x["country_full_name1"]] if ( x["country_full_name1"] in country_gdp_per_person_dict) else np.nan, axis=1)
    pairs["country_gdp2_per_person"] = pairs.apply(lambda x: country_gdp_per_person_dict[x["country_full_name2"]] if (x["country_full_name2"] in country_gdp_per_person_dict) else np.nan, axis=1)

    indexes_stats = pd.read_csv("indexes/indexes_stats.csv")
    indexes_stats = indexes_stats.drop(indexes_stats.columns[[0]], axis = 1)


    # # this was for filtering out the pairs whose languages we didn't cover, but the number is very limited, just 10 out of 20miilions.
    # pairs = pd.read_csv(args.input_file)
    # pairs["lang_valid"] = pairs.apply(lambda x: 1 if ((x["language1"] in lang_dict) and (x["a2_language"] in lang_dict)) else 0, axis=1)
    # pairs = pairs.loc[pairs["lang_valid"] == 1]
    # pairs_dict = {key: pairs[key].to_list() for key in pairs.keys()}

    # load geographic info
    country_geography_list = pd.read_csv("country_info/country_geo_location.csv")

    country_alpha3_code = country_geography_list["Alpha-3 code"].to_list()
    country_full_name = country_geography_list["Country"].to_list()
    country_latitude = country_geography_list["Latitude (average)"].to_list()
    country_longitude = country_geography_list["Longitude (average)"].to_list()
    country_alpha3_geography = defaultdict(dict)
    for i in range(len(country_alpha3_code)):
        country_alpha3_geography[country_alpha3_code[i]] = {"country_full_name": country_full_name[i],
                                                            "country_latitude": country_latitude[i],
                                                            "country_longitude": country_longitude[i]}


    country_official_languages = pd.read_csv("country_info/country_official_language.csv")
    country_official_languages["Country"] = country_official_languages.apply(lambda x: x['Country'].replace("\xa0", ""), axis=1)
    country_official_languages["Official language"] = country_official_languages.apply(lambda x: x['Official language'].replace("\xa0", "").replace("\u2028", "\n").split("\n"), axis=1)
    country_official_languages["covered"] = country_official_languages.apply(lambda x: any([lang in lang_full_name_dict for lang in x["Official language"]]), axis=1)

    country_official_languages = pd.merge(country_geography_list, country_official_languages, how='left', on='Country')
    country_official_languages = country_official_languages[['Country', 'Alpha-3 code', 'Official language', 'covered']]
    country_official_languages = country_official_languages.dropna(subset=["covered"])
    country_official_languages = country_official_languages[country_official_languages['covered'] == True]
    covered_country = {key: 1 for key in country_official_languages["Alpha-3 code"].to_list()}

    pairs = pairs[pairs["main_country1"].isin(covered_country)]
    pairs = pairs[pairs["main_country2"].isin(covered_country)]

    # args.cluster can't be used at the same time as args.opt == "rm" or args.opt == "rm_intra"
    if args.cluster_sim:
        pairs = pairs.dropna(subset=["main_country1", "main_country2"])


        all_country_cls_df = pd.read_csv("network_data/country_cluster/country_cluster.csv")
        all_country_cls_dict = all_country_cls_df.to_dict()
        all_country_cls_list = defaultdict(list)
        for i in range(all_country_cls_df.shape[0]):
            for cls in all_country_cls_dict:
                if cls == "Unnamed: 0":
                    alpha3 = all_country_cls_dict[cls][i]
                else:
                    all_country_cls_list[alpha3].append(all_country_cls_dict[cls][i])

        entropy_dict = {}
        jensenshannon_dict = defaultdict(dict)
        jensenshannon_for_save_dict = defaultdict(dict)
        jensenshannon_for_print_dict = {}
        for cur_alpha3_1 in all_country_cls_list:
            for cur_alpha3_2 in all_country_cls_list:
                try:
                    jensenshannon_dict[cur_alpha3_1][cur_alpha3_2] = 1 - distance.jensenshannon(all_country_cls_list[cur_alpha3_1], all_country_cls_list[cur_alpha3_2])
                    jensenshannon_for_save_dict[country_alpha3_to_full_name[cur_alpha3_1]][country_alpha3_to_full_name[cur_alpha3_2]] = 1 - distance.jensenshannon(all_country_cls_list[cur_alpha3_1], all_country_cls_list[cur_alpha3_2])
                    # filtered those self-loop and duplicate of (alpha3_1, alpha3_2) and (alpha3_2, alpha3_1)
                    if cur_alpha3_1 < cur_alpha3_2:
                        jensenshannon_for_print_dict[(cur_alpha3_1,cur_alpha3_2)] = 1 - distance.jensenshannon(all_country_cls_list[cur_alpha3_1], all_country_cls_list[cur_alpha3_2])
                except:
                    pass
        for cur_alpha3_1 in all_country_cls_list:
            entropy_dict[cur_alpha3_1] = 1 - entropy(all_country_cls_list[cur_alpha3_1]) / np.log(len(all_country_cls_list[cur_alpha3_1]))

        def switch_cls_sim(x):
            if x["main_country1"] == x["main_country2"]:
                try:
                    cls_sim = entropy_dict[x["main_country1"]]
                except:
                    cls_sim = -1
            else:
                try:
                    cls_sim = jensenshannon_dict[x["main_country1"]][x["main_country2"]]
                except:
                    cls_sim = -1
            return cls_sim

        # filtered those pairs whose pair might not match
        pairs["similarity"] = pairs.apply(lambda x: switch_cls_sim(x), axis=1)
        pairs = pairs[pairs["similarity"] != -1]

        most_similar_countries = dict(sorted(entropy_dict.items(), key=operator.itemgetter(1), reverse=True)[:10])
        most_dissimilar_countries = dict(sorted(entropy_dict.items(), key=operator.itemgetter(1))[:10])
        most_similar_country_pairs = dict(sorted(jensenshannon_for_print_dict.items(), key=operator.itemgetter(1), reverse=True)[:10])
        most_dissimilar_country_pairs = dict(sorted(jensenshannon_for_print_dict.items(), key=operator.itemgetter(1))[:10])

        print("countries with highest cluster-based similarity:")
        for k,v in most_similar_countries.items():
            if k in country_alpha3_to_full_name:
                print(f"{country_alpha3_to_full_name[k]}: {v}")
        print()

        print("countries with lowest cluster-based similarity:")
        for k,v in most_dissimilar_countries.items():
            if k in country_alpha3_to_full_name:
                print(f"{country_alpha3_to_full_name[k]}: {v}")
        print()

        print("country pairs with highest cluster-based similarity:")
        for k,v in most_similar_country_pairs.items():
            cur_alpha3_1, cur_alpha3_2 = k
            if (cur_alpha3_1 in country_alpha3_to_full_name) and (cur_alpha3_2 in country_alpha3_to_full_name):
                print(f"country1: {country_alpha3_to_full_name[cur_alpha3_1]} country2: {country_alpha3_to_full_name[cur_alpha3_2]}", "   similarity:",v)
        print()

        print("country pairs with lowest cluster-based similarity:")
        for k,v in most_dissimilar_country_pairs.items():
            cur_alpha3_1, cur_alpha3_2 = k
            if (cur_alpha3_1 in country_alpha3_to_full_name) and (cur_alpha3_2 in country_alpha3_to_full_name):
                print(f"country1: {country_alpha3_to_full_name[cur_alpha3_1]} country2: {country_alpha3_to_full_name[cur_alpha3_2]}", "   similarity:",v)
        print()
        
        jensenshannon_for_save_dict_df = pd.DataFrame(jensenshannon_for_save_dict)
        jensenshannon_for_save_dict_df.to_csv("network_data/country_cluster/country_cls_sim.csv")

    if args.stat:
        globstring = "indexes/*-wiki-stat-v2.index"
        # globstring = "indexes/zh-wiki-stat-v2.index"
        stat_list = []

        stat_files = sorted(glob.glob(globstring), reverse=True)
        for file in stat_files:
            lang = file.split("/")[-1][:2]
            cur_stat = pd.read_csv(file, sep="\t", names=["relatime", "media_id"])
            cur_stat_count = dict(cur_stat.groupby(["relatime", "media_id"]).size())

            cur_stat = cur_stat.drop_duplicates()
            cur_stat["lang"] = cur_stat.apply(lambda x: lang, axis=1)
            cur_stat["count"] = cur_stat.apply(lambda x: cur_stat_count[(x["relatime"], x["media_id"])], axis=1)

            stat_list.append(cur_stat)
        stat_df = pd.concat(stat_list)

        # match media_id to country info

        ''' load country list '''
        # the forms of "pub_country" and "about_country" ara different
        # pub_country is shortcut of the name, while about_country is the full name
        # the shortcut can be matched to the geonames table - "alpha3" attribute, and full name can be matched to the geonames table - "label" attribute
        # e.g. pub_country = "USA", and about_country = "United States"

        country_outlets_list = pd.read_csv("country_info/integrated_country.csv")
        # country_outlets_url = country_outlets_list["url"].to_list()
        country_outlets_id = country_outlets_list["media_id"].to_list()
        country_outlets_pub_country = country_outlets_list["pub_country"].to_list()
        country_outlets_about_country = country_outlets_list["about_country"].to_list()
        country_outlets_id_pub_country = {}
        country_outlets_id_about_country = {}
        for i in range(len(country_outlets_id)):
            country_outlets_id_pub_country[country_outlets_id[i]] = country_outlets_pub_country[i]
            country_outlets_id_about_country[country_outlets_id[i]] = country_outlets_about_country[i]

        '''load geographic location'''
        country_geography_list = pd.read_csv("country_info/country_geo_location.csv")
        country_alpha3_code = country_geography_list["Alpha-3 code"].to_list()
        country_full_name = country_geography_list["Country"].to_list()
        country_latitude = country_geography_list["Latitude (average)"].to_list()
        country_longitude = country_geography_list["Longitude (average)"].to_list()
        country_alpha3_geography = defaultdict(dict)
        for i in range(len(country_alpha3_code)):
            country_alpha3_geography[country_alpha3_code[i]] = {"country_full_name": country_full_name[i],
                                                                "country_latitude": country_latitude[i],
                                                                "country_longitude": country_longitude[i]}


        ''' load bias list '''
        bias_outlets_list = pd.read_csv("integrated_bias.csv")
        bias_outlets_url = bias_outlets_list["link"].to_list()
        bias_outlets_bias = bias_outlets_list["bias"].to_list()
        bias_outlets_country = bias_outlets_list["country"].to_list()
        bias_outlets_url_bias = {}
        bias_outlets_url_country = {}
        for i in range(len(bias_outlets_url)):
            bias_outlets_url[i] = unify_url(bias_outlets_url[i])
            if isinstance(bias_outlets_country[i], str):
                bias_outlets_country[i] = bias_outlets_country[i].replace("United States", "USA").split(" (")[0]
            bias_outlets_url_bias[bias_outlets_url[i]] = bias_outlets_bias[i]
            bias_outlets_url_country[bias_outlets_url[i]] = bias_outlets_country[i]

        '''load democracy index'''
        country_democracy_index_list = pd.read_csv("bias_dataset/2019_democracy_index/2019_democracy_index.csv")
        country_democracy_index_list = country_democracy_index_list[country_democracy_index_list["year"] == 2019]
        country_dem_list_country_full_name = country_democracy_index_list["country"].to_list()
        country_dem_list_democracy_index = country_democracy_index_list["eiu"].to_list()
        country_democracy_index = {}
        for i in range(len(country_dem_list_country_full_name)):
            country_democracy_index[country_dem_list_country_full_name[i]] = country_dem_list_democracy_index[i]

        stat_df['lang_family'] = stat_df.apply(lambda x: LANG_FAMILY[x["lang"]],axis=1)
        stat_df['pub_country'] = stat_df.apply(lambda x: country_outlets_id_pub_country[x["media_id"]] if x["media_id"] in country_outlets_id_pub_country else "",axis=1)
        stat_df['about_country'] = stat_df.apply(lambda x: country_outlets_id_about_country[x["media_id"]] if x["media_id"] in country_outlets_id_about_country else "", axis=1)
        # didn't include bias country here so the main country here is kind of different from it when processing the network inference
        stat_df['main_country'] = stat_df.apply(lambda x: x['pub_country'] if x['pub_country'] != "" else x['about_country'], axis=1)

        stat_df = stat_df.dropna(subset=["main_country"])
        stat_df = stat_df[stat_df["main_country"] != ""]

        stat_df['country_full_name'] = stat_df.apply(lambda x: country_alpha3_geography[x["main_country"]]["country_full_name"] if x['main_country'] in country_alpha3_geography else "", axis=1)
        stat_df['latitude'] = stat_df.apply(lambda x: country_alpha3_geography[x["main_country"]]["country_latitude"] if x['main_country'] in country_alpha3_geography else "", axis=1)
        stat_df['longitude'] = stat_df.apply(lambda x: country_alpha3_geography[x["main_country"]]["country_longitude"] if x['main_country'] in country_alpha3_geography else "", axis=1)
        stat_df['democracy_index'] = stat_df.apply(lambda x: country_democracy_index[x["country_full_name"]] if x['country_full_name'] in country_democracy_index else "", axis=1)

        stat_df.to_csv("indexes/indexes_stats.csv")

    if args.option == "centrality":
        # country_cls_sim = pd.read_csv("network_data/country_cluster/country_cls_sim.csv")
        country_cls_sim = pd.read_csv("network_data/country_cluster/server/country_cls_sim.csv")

        country_cls_sim = country_cls_sim.drop(country_cls_sim.columns[0], axis=1)
        country_cls_sim.index = country_cls_sim.columns.to_list()
        # country_cls_sim = country_cls_sim.applymap(lambda x: 10*(1-x))
        country_graph = igraph.Graph.Weighted_Adjacency((country_cls_sim.values > 0).tolist(), mode='undirected', loops=False)
        country_graph.vs['label'] = country_cls_sim.columns
        country_graph.vs['size'] = [0.05 for i in country_cls_sim.columns]
        country_graph.es['weight'] = country_cls_sim.values[country_cls_sim.values.nonzero()]
        # country_graph.es['width'] = country_cls_sim.values[country_cls_sim.values.nonzero()]
        #community detection

        # i = country_graph.community_infomap()
        i = country_graph.community_edge_betweenness()
        ii = i.as_clustering()
        pal = igraph.drawing.colors.ClusterColoringPalette(len(ii))
        country_graph.vs['color'] = pal.get_many(ii.membership)



        fig, ax = plt.subplots()
        igraph.plot(country_graph, target=ax)
        # igraph.plot(country_graph, layout=country_graph.layout("kk"), target=ax)
        # igraph.plot(country_graph, layout=country_graph.layout_random(), target=ax)
        # igraph.plot(country_graph, layout=country_graph.layout_reingold_tilford(root=[2]), target=ax)
        plt.show()
        
    # save country level data into files
    if args.option == "save":
        # load filter-out article pairs data
        geo_index_art_data = defaultdict(int)
        for pair in indexes_stats.itertuples():
            cur_main_country = pair[9]
            cur_art_count = pair[4]
            geo_index_art_data[cur_main_country] += cur_art_count

        with open("indexes/art_num_b4_filter.json", 'w') as f:
            json.dump(geo_index_art_data, f)

    if args.option == "gm":
        # align similarity value to fit the visualization
        pairs["similarity"] = pairs.apply(lambda x: (5 - x["similarity"]), axis=1)

        '''load democracy index'''
        country_democracy_index_list = pd.read_csv("bias_dataset/2019_democracy_index/2019_democracy_index.csv")
        country_democracy_index_list = country_democracy_index_list[country_democracy_index_list["year"] == 2019]
        country_democracy_index_list.rename(columns={'country': 'Country'}, inplace=True)
        country_democracy_index_list["Country"] = country_democracy_index_list["Country"].apply(
            lambda x: "United States" if x == "US" else x)
        country_democracy_index_list["Country"] = country_democracy_index_list["Country"].apply(
            lambda x: "United Kingdom" if x == "UK" else x)

        country_unitary_state_list = pd.read_csv("country_info/country_unitary_state.csv")

        country_alpha3_full_name_list = pd.merge(country_democracy_index_list, country_geography_list, on='Country')
        country_alpha3_full_name_list = pd.merge(country_alpha3_full_name_list, country_unitary_state_list,
                                                 on='Country')
        country_alpha3_unitary_dict = {}
        for country in country_alpha3_full_name_list.itertuples():
            cur_alpha3 = country[4]
            cur_unitary = country[9]

            country_alpha3_unitary_dict[cur_alpha3] = cur_unitary

        ''' geographic visualization '''
        # in this visualization, drops pairs don't have a specific location
        shown_edge_num = 80
        geo_pairs = pairs.dropna(subset=["longitude1", "longitude2", "latitude1", "latitude2"])

        # filtering the intra-country pairs whose country official language in not covered in the 10 languages of our annotation
        country_official_languages = pd.read_csv("country_info/country_official_language.csv")
        country_official_languages["Country"] = country_official_languages.apply(lambda x: x['Country'].replace("\xa0", ""), axis=1)
        country_official_languages["Official language"] = country_official_languages.apply(lambda x: x['Official language'].replace("\xa0", "").replace("\u2028", "\n").split("\n"), axis=1)
        country_official_languages["covered"] = country_official_languages.apply(lambda x: any([lang in lang_full_name_dict for lang in x["Official language"]]), axis=1)

        country_official_languages = pd.merge(country_geography_list, country_official_languages, how='left', on='Country')
        country_official_languages = country_official_languages[['Country', 'Alpha-3 code', 'Official language', 'covered']]
        country_official_languages = country_official_languages.dropna(subset=["covered"])
        country_official_languages = country_official_languages[country_official_languages['covered'] == True]
        covered_country = {key: 1 for key in country_official_languages["Alpha-3 code"].to_list()}

        print("we covered ", len(covered_country), "countries in these 10 language")

        geo_pairs_dict = {key: geo_pairs[key].to_list() for key in geo_pairs.keys()}

        geo_node_loc = {}  # node's size (similarity)
        geo_node_dem = {}  # node's color (democracy index)
        geo_node_sim = defaultdict(list)
        geo_edge_loc = {}
        geo_edge_num = collections.Counter()  # we count the edge number to select the top links with the most article pairs
        geo_edge_sim = defaultdict(list)  # edge's size (similarity)

        # filter the countries have less than certain number of media to mitigating the variance
        min_media_num = 10
        country_outlet_name_list = defaultdict(dict)

        geo_edge_data = {}
        for i in range(len(geo_pairs)):
            # node case
            lon1 = geo_pairs_dict["longitude1"][i]
            lon2 = geo_pairs_dict["longitude2"][i]
            lat1 = geo_pairs_dict["latitude1"][i]
            lat2 = geo_pairs_dict["latitude2"][i]

            cur_country1 = geo_pairs_dict["main_country1"][i]
            cur_country2 = geo_pairs_dict["main_country2"][i]

            if (lon1 == lon2) and (lat1 == lat2):
                if cur_country1 in covered_country:
                    geo_node_loc[cur_country1] = (lon1, lat1)
                    geo_node_dem[cur_country1] = geo_pairs_dict["democracy_index1"][i]
                    geo_node_sim[cur_country1].append(geo_pairs_dict["similarity"][i])
            else:
                if cur_country1 in covered_country and cur_country2 in covered_country:
                    if cur_country1 < cur_country2:
                        country_pair = cur_country1 + "_" + cur_country2
                        geo_edge_data[country_pair] = (cur_country1, cur_country2)
                        geo_edge_loc[country_pair] = ((lon1, lat1), (lon2, lat2))
                    else:
                        country_pair = cur_country2 + "_" + cur_country1
                        geo_edge_data[country_pair] = (cur_country2, cur_country1)
                        geo_edge_loc[country_pair] = ((lon2, lat2), (lon1, lat1))

                    geo_edge_num[country_pair] += 1
                    geo_edge_sim[country_pair].append(geo_pairs_dict["similarity"][i])

            outlet_name1 = geo_pairs_dict["media_name1"][i]
            outlet_name2 = geo_pairs_dict["media_name2"][i]
            if len(country_outlet_name_list[cur_country1]) < min_media_num:
                country_outlet_name_list[cur_country1][outlet_name1] = 1
            if len(country_outlet_name_list[cur_country2]) < min_media_num:
                country_outlet_name_list[cur_country2][outlet_name2] = 1

        intra_country_pair_num = defaultdict(dict)
        for alpha3 in geo_node_sim:
            intra_country_pair_num[country_alpha3_geography[alpha3]["country_full_name"]]["pair_num"] = len(
                geo_node_sim[alpha3])
            intra_country_pair_num[country_alpha3_geography[alpha3]["country_full_name"]]["avg_sim"] = np.mean(
                geo_node_sim[alpha3])
        with open("indexes/pair_intra_country.json", 'w') as f:
            json.dump(intra_country_pair_num, f)

        geo_index_art_data = defaultdict(int)
        for pair in indexes_stats.itertuples():
            cur_main_country = pair[9]
            cur_alpha3 = pair[8]
            cur_art_count = pair[4]

            geo_index_art_data[cur_main_country] += cur_art_count

        geo_covered_art_data = defaultdict(int)
        for pair in indexes_stats.itertuples():
            cur_main_country = pair[9]
            cur_alpha3 = pair[8]
            cur_art_count = pair[4]
            if cur_alpha3 in covered_country:
                geo_covered_art_data[cur_main_country] += cur_art_count

        # plot global articles of each country
        a = list(geo_covered_art_data.keys())
        b = list(geo_covered_art_data.values())
        print("total articles we covered: ", sum(b))
        geo = (
            Map(init_opts=opts.InitOpts(theme='essos', width="800px", height='550px'))  # 图表大小
                .add("", [list(z) for z in zip(a, b)], "world", is_map_symbol_show=False)
                .set_series_opts(label_opts=opts.LabelOpts(is_show=False))  # 标签不显示(国家名称不显示)
                .set_global_opts(
                title_opts=opts.TitleOpts(title="Article numbers of each country", subtitle='article numbers'),
                # 主标题与副标题名称
                visualmap_opts=opts.VisualMapOpts(max_=1500000),  # 值映射最大值
            )
        )
        geo.render("geo_art.html")

        # find the top k edge with most article pairs
        top_k_edges = geo_edge_num.most_common(shown_edge_num)

        print("top country pairs with the most article pairs:")
        for cur_geo_country_pair in geo_edge_num.most_common(30):
            print(cur_geo_country_pair)

        # plot global sim
        geo = Geo(init_opts=opts.InitOpts(width='1500px', height='900px', ))
        geo.add_schema(maptype="world")

        unitary_type_vis = {"unitary republics": 10, "federalism": 5}

        # geo = Map(init_opts=opts.InitOpts(width='1500px', height='900px', ))

        # Add coordinate points, add names and longitude and latitude
        for country in country_alpha3_geography:
            geo.add_coordinate(name=country, longitude=country_alpha3_geography[country]["country_longitude"],
                               latitude=country_alpha3_geography[country]["country_latitude"])

        geo_node_sim_data = [(country, trans_node_sim(np.mean(geo_node_sim[country]))) for country in geo_node_loc]
        geo_node_loc_key = [country for country in geo_node_loc]
        for i in range(len(geo_node_sim_data)):
            if len(country_outlet_name_list[geo_node_loc_key[i]]) < min_media_num:
                continue
            if (geo_node_sim_data[i][0] in country_alpha3_unitary_dict) and (
                    country_alpha3_unitary_dict[geo_node_sim_data[i][0]] in unitary_type_vis):
                geo.add("intra-country", [
                    (geo_node_sim_data[i][0], unitary_type_vis[country_alpha3_unitary_dict[geo_node_sim_data[i][0]]])],
                        symbol_size=geo_node_sim_data[i][-1])
            else:
                geo.add("intra-country", [(geo_node_sim_data[i][0], 0)], symbol_size=geo_node_sim_data[i][-1])

                # Draw edges
        top_k_edges_tuple = [cur_edge[0].split("_") for cur_edge in top_k_edges]
        top_k_edge_width = [trans_edge_sim(np.mean(geo_edge_sim[cur_edge[0]])) for cur_edge in
                            top_k_edges]  # should be similarity

        for i in range(len(top_k_edges)):
            geo.add("", [top_k_edges_tuple[i]],
                    type_=GeoType.LINES,
                    # type_=ChartType.LINES,
                    symbol_size=0,
                    effect_opts=opts.EffectOpts(symbol=SymbolType.RECT, symbol_size=0, color="grey"),
                    # is_polyline=True,
                    linestyle_opts=opts.LineStyleOpts(curve=0.2, opacity=0.05, color="orange",
                                                      width=0.1 * top_k_edge_width[i]), )

        geo.set_series_opts(label_opts=opts.LabelOpts(is_show=False))
        geo.set_global_opts(visualmap_opts=opts.VisualMapOpts(max_=10), title_opts=opts.TitleOpts(title="mygeo"))
        geo.render("geo_sim.html")

        # plot global article pair number
        geo = Geo(init_opts=opts.InitOpts(width='1500px', height='900px', ))
        geo.add_schema(maptype="world")

        # Add coordinate points, add names and longitude and latitude
        for country in country_alpha3_geography:
            geo.add_coordinate(name=country, longitude=country_alpha3_geography[country]["country_longitude"],
                               latitude=country_alpha3_geography[country]["country_latitude"])

        geo_node_sim_data = [(country, math.pow(2000 * np.sqrt(0.001 * len(geo_node_sim[country])), 1 / 3)) for country
                             in geo_node_loc]
        geo_node_loc_key = [country for country in geo_node_loc]
        for i in range(len(geo_node_sim_data)):
            if len(country_outlet_name_list[geo_node_loc_key[i]]) < min_media_num:
                continue
            geo.add("intra-country", [geo_node_sim_data[i]], symbol_size=geo_node_sim_data[i][-1])

        # Draw edges
        top_k_edges_tuple = [cur_edge[0].split("_") for cur_edge in top_k_edges]
        top_k_edge_width = [np.sqrt(0.001 * len(geo_edge_sim[cur_edge[0]])) for cur_edge in
                            top_k_edges]  # should be similarity

        inter_country_pair = defaultdict(dict)
        for country_pair in list(geo_edge_sim.keys()):
            cur_alpha3_1, cur_alpha3_2 = country_pair.split("_")
            cur_country1 = country_alpha3_geography[cur_alpha3_1]["country_full_name"]
            cur_country2 = country_alpha3_geography[cur_alpha3_2]["country_full_name"]

            inter_country_pair[country_pair]["country1_alpha3"] = cur_alpha3_1
            inter_country_pair[country_pair]["country2_alpha3"] = cur_alpha3_2
            inter_country_pair[country_pair]["country1"] = cur_country1
            inter_country_pair[country_pair]["country2"] = cur_country2
            inter_country_pair[country_pair]["pair_num"] = len(geo_edge_sim[country_pair])
            inter_country_pair[country_pair]["avg_sim"] = np.mean(geo_edge_sim[country_pair])
        with open("indexes/pair_inter_country.json", 'w') as f:
            json.dump(inter_country_pair, f)

        for i in range(len(top_k_edges)):
            geo.add("", [top_k_edges_tuple[i]],
                    type_=GeoType.LINES,
                    # type_=ChartType.LINES,
                    symbol_size=0,
                    effect_opts=opts.EffectOpts(symbol=SymbolType.RECT, symbol_size=0, color="grey"),
                    # is_polyline=True,
                    linestyle_opts=opts.LineStyleOpts(curve=0.2, opacity=0.05, color="Indigo",
                                                      width=2 * top_k_edge_width[i]), )

        geo.set_series_opts(label_opts=opts.LabelOpts(is_show=False))
        geo.set_global_opts(visualmap_opts=opts.VisualMapOpts(max_=10), title_opts=opts.TitleOpts(title="mygeo"))
        geo.render("geo_pair_num.html")

        # plot global article pair filter-in fraction
        geo = Geo(init_opts=opts.InitOpts(width='1500px', height='900px', ))
        geo.add_schema(maptype="world")

        # Add coordinate points, add names and longitude and latitude
        for country in country_alpha3_geography:
            geo.add_coordinate(name=country, longitude=country_alpha3_geography[country]["country_longitude"],
                               latitude=country_alpha3_geography[country]["country_latitude"])

        geo_node_sim_data = [(country, math.pow(100000000000 * len(geo_node_sim[country]) / (
                    geo_index_art_data[country] * (geo_index_art_data[country] - 1)), 1 / 3)) for country in
                             geo_node_loc]
        geo_node_loc_key = [country for country in geo_node_loc]
        for i in range(len(geo_node_sim_data)):
            if len(country_outlet_name_list[geo_node_loc_key[i]]) < min_media_num:
                continue
            geo.add("intra-country", [geo_node_sim_data[i]], symbol_size=geo_node_sim_data[i][-1])

        # Draw edges
        top_k_edges_tuple = [cur_edge[0].split("_") for cur_edge in top_k_edges]
        top_k_edge_width = [50000000 * top_k_edges[i][1] / (
                    geo_index_art_data[top_k_edges_tuple[i][0]] * geo_index_art_data[top_k_edges_tuple[i][1]]) for i in
                            range(len(top_k_edges))]  # should be similarity

        for i in range(len(top_k_edges)):
            geo.add("", [top_k_edges_tuple[i]],
                    type_=GeoType.LINES,
                    # type_=ChartType.LINES,
                    symbol_size=0,
                    effect_opts=opts.EffectOpts(symbol=SymbolType.RECT, symbol_size=0, color="grey"),
                    # is_polyline=True,
                    linestyle_opts=opts.LineStyleOpts(curve=0.2, opacity=0.05, color="orange",
                                                      width=0.1 * top_k_edge_width[i]), )

        geo.set_series_opts(label_opts=opts.LabelOpts(is_show=False))
        geo.set_global_opts(visualmap_opts=opts.VisualMapOpts(max_=10), title_opts=opts.TitleOpts(title="mygeo"))
        geo.render("geo_pair_frac.html")

    #democracy analysis
    if args.option == "da":
        '''load democracy index'''
        country_democracy_index_list = pd.read_csv("bias_dataset/2019_democracy_index/2019_democracy_index.csv")
        country_democracy_index_list = country_democracy_index_list[country_democracy_index_list["year"] == 2019]
        country_democracy_index_list.rename(columns={'country': 'Country'}, inplace=True)

        country_unitary_state_list = pd.read_csv("country_info/country_unitary_state.csv")

        country_alpha3_full_name_list = pd.merge(country_democracy_index_list, country_geography_list, on='Country')
        country_alpha3_full_name_list = pd.merge(country_alpha3_full_name_list, country_unitary_state_list, on='Country')

        country_alpha3_full_name_unitary_dict = {}
        country_alpha3_full_name_unitary_dict["unitary_republic"] = country_alpha3_full_name_list[country_alpha3_full_name_list['Unitary'] == 'unitary republics']
        country_alpha3_full_name_unitary_dict["unitary_monarchies"] = country_alpha3_full_name_list[country_alpha3_full_name_list['Unitary'] == 'unitary monarchies']
        country_alpha3_full_name_unitary_dict["federalism"] = country_alpha3_full_name_list[country_alpha3_full_name_list['Unitary'] == 'federalism']

        country_alpha3_dem_dict = { country[5]: country[3] for country in country_alpha3_full_name_list.itertuples()}
        country_alpha3_dem_unitary_dict = {}
        country_alpha3_dem_unitary_dict["unitary_republic"] = { country[5]: country[3] for country in country_alpha3_full_name_unitary_dict["unitary_republic"].itertuples()}
        country_alpha3_dem_unitary_dict["unitary_monarchies"] = { country[5]: country[3] for country in country_alpha3_full_name_unitary_dict["unitary_monarchies"].itertuples()}
        country_alpha3_dem_unitary_dict["federalism"] = { country[5]: country[3] for country in country_alpha3_full_name_unitary_dict["federalism"].itertuples()}

        # first draw a scatter plot for similarity of each country, x is the democracy index

        country_pairs = pairs.dropna(subset=["main_country1", "main_country2", "democracy_index1", "democracy_index2"])
        country_pairs['country_type'] = country_pairs.apply(lambda x: "intra-country" if (x["main_country1"] == x["main_country2"]) else "inter-country", axis=1)
        country_sim_dict = defaultdict(list)
        # country_pairs.apply(lambda x: country_sim_dict[x["main_country1"]].append(trans_node_sim(x["similarity"])) if x["main_country1"] == x["main_country2"] else 0, axis=1)
        country_pairs.apply(lambda x: country_sim_dict[x["main_country1"]].append(x["similarity"]) if x["main_country1"] == x["main_country2"] else 0, axis=1)

        # loading index data for statistics
        indexes_stats_dem_idx = indexes_stats.drop(["relatime"], axis=1)
        indexes_stats_dem_idx = indexes_stats_dem_idx.dropna(subset=["main_country", "democracy_index"])

        indexes_stats_dem_idx["democracy_index_class"] = indexes_stats_dem_idx.apply(lambda x: math.ceil(x["democracy_index"] * 2.5) / 2.5, axis=1)
        indexes_stats_dem_idx["unitary_type"] = indexes_stats_dem_idx.apply(lambda x: find_unitary_type(x["main_country"], country_alpha3_dem_unitary_dict), axis=1)
        indexes_stats_group_by_dem_idx_count = dict(indexes_stats_dem_idx.groupby(["main_country"])["count"].sum())

        indexes_stats_dem_idx["dem_count"] = indexes_stats_dem_idx.apply(lambda x: indexes_stats_group_by_dem_idx_count[x["main_country"]], axis=1)
        indexes_stats_dem_idx = indexes_stats_dem_idx.drop(["lang", "media_id", "count", "pub_country", "about_country"], axis=1)
        indexes_stats_dem_idx = indexes_stats_dem_idx.drop_duplicates()

        indexes_stats_dem_idx_country = pd.DataFrame()
        indexes_stats_dem_idx_unitary_type = pd.DataFrame()
        indexes_stats_dem_idx_country_dict = {}
        indexes_stats_dem_idx_country["democracy_index_class"] = indexes_stats_dem_idx.groupby('democracy_index_class').agg({'dem_count': list}).reset_index().apply(lambda d: d.democracy_index_class, axis=1)
        indexes_stats_dem_idx_country["intra_country"] = indexes_stats_dem_idx.groupby('democracy_index_class').agg({'dem_count': list}).reset_index().apply(lambda d: dem_class_count(d['dem_count'], "intra_country"), axis=1)
        indexes_stats_dem_idx_country["inter_country"] = indexes_stats_dem_idx.groupby('democracy_index_class').agg({'dem_count': list}).reset_index().apply(lambda d: dem_class_count(d['dem_count'], "inter_country"), axis=1)
        indexes_stats_dem_idx_unitary_type["democracy_index_class"] = indexes_stats_dem_idx.groupby(['democracy_index_class', 'unitary_type']).agg({'dem_count': list}).reset_index().apply(lambda d: d.democracy_index_class, axis=1)
        indexes_stats_dem_idx_unitary_type["unitary_type"] = indexes_stats_dem_idx.groupby(['democracy_index_class', 'unitary_type']).agg({'dem_count': list}).reset_index().apply(lambda d: d.unitary_type, axis=1)
        indexes_stats_dem_idx_unitary_type["intra_country"] = indexes_stats_dem_idx.groupby(['democracy_index_class', 'unitary_type']).agg({'dem_count': list}).reset_index().apply(lambda d: dem_class_count(d['dem_count'], "intra_country"), axis=1)
        for dem_idx_country_count in indexes_stats_dem_idx_country.itertuples():
            cur_democracy_index_class = dem_idx_country_count[1]
            cur_intra_country_count = dem_idx_country_count[2]
            cur_inter_country_count = dem_idx_country_count[3]

            indexes_stats_dem_idx_country_dict[(cur_democracy_index_class, "intra-country")] = cur_intra_country_count
            indexes_stats_dem_idx_country_dict[(cur_democracy_index_class, "inter-country")] = cur_inter_country_count

        for dem_idx_unitary_count in indexes_stats_dem_idx_unitary_type.itertuples():
            cur_democracy_index_class = dem_idx_unitary_count[1]
            cur_unitary_type = dem_idx_unitary_count[2]
            cur_intra_unitary_count = dem_idx_unitary_count[3]

            indexes_stats_dem_idx_country_dict[(cur_democracy_index_class, cur_unitary_type)] = cur_intra_unitary_count
        # for test

        min_pair_num = 50

        country_dem_for_present = []
        country_sim_for_present = []
        country_dem_unitary_for_present = defaultdict(list)
        country_sim_unitary_for_present = defaultdict(list)

        for country in country_sim_dict:
            if len(country_sim_dict[country]) >= min_pair_num:
                try:
                    country_dem_for_present.append(country_alpha3_dem_dict[country])
                    country_sim_for_present.append(np.mean(country_sim_dict[country]))
                except:
                    pass
                try:
                    for unitary_type in country_alpha3_dem_unitary_dict:
                        if country in country_alpha3_dem_unitary_dict[unitary_type]:
                            country_dem_unitary_for_present[unitary_type].append(country_alpha3_dem_dict[country])
                            country_sim_unitary_for_present[unitary_type].append(np.mean(country_sim_dict[country]))
                except:
                    pass

        dem_sim_scatter_dict = {"Democracy Index":country_dem_for_present, "Similarity":country_sim_for_present}
        dem_sim_scatter_df = pd.DataFrame(dem_sim_scatter_dict)
        fig = sns.lmplot(x="Democracy Index", y="Similarity", data=dem_sim_scatter_df, order=3, ci=95)
        fig.savefig(f"{output_dir}/dem_sim_scatter.png")


        # plot scatter according to their political system type
        unitary_color = {"unitary_republic":"red", "unitary_monarchies":"blue", "federalism":"green"}
        fig = plt.figure(figsize=(10, 8))
        for unitary_type in country_alpha3_dem_unitary_dict:
            plt.scatter(country_dem_unitary_for_present[unitary_type], country_sim_unitary_for_present[unitary_type], c= unitary_color[unitary_type])
        plt.xticks(size=20)
        plt.yticks(size=20)
        plt.xlabel('Democracy Index', fontsize=32)
        plt.ylabel('Similarity', fontsize=32)
        plt.legend(["unitary republic", "unitary monarchies", "federalism"],fontsize= 20)
        # plt.show()
        plt.savefig(f"{output_dir}/dem_sim_unitary_scatter.png")

        # plot intra-country similarity of countries with different democracy index
        country_dem_idx_pairs = country_pairs.loc[country_pairs["main_country1"] == country_pairs["main_country2"]]

        # country_dem_idx_pairs["democracy_index_class"] = country_dem_idx_pairs.apply(lambda x: math.ceil(x["democracy_index1"]/2) * 2, axis=1)
        # country_dem_idx_pairs["democracy_index_class"] = country_dem_idx_pairs.apply(lambda x: math.ceil(x["democracy_index1"]), axis=1)
        # country_dem_idx_pairs["democracy_index_class"] = country_dem_idx_pairs.apply(lambda x: math.ceil(x["democracy_index1"] * 2) / 2, axis=1)
        country_dem_idx_pairs["democracy_index_class"] = country_dem_idx_pairs.apply(lambda x: math.ceil(x["democracy_index1"] * 2.5) / 2.5, axis=1)
        country_dem_idx_pairs["unitary_type1"] = country_dem_idx_pairs.apply(lambda x: find_unitary_type(x["main_country1"], country_alpha3_dem_unitary_dict), axis=1)

        # plot overall intra-country similarity with democracy index
        min_pair_num = 1000

        country_pairs_group_by_dem_idx_mean = dict(country_dem_idx_pairs.groupby(["democracy_index_class", "country_type"])["similarity"].mean())
        country_pairs_group_by_dem_idx_count = dict(country_dem_idx_pairs.groupby(["democracy_index_class", "country_type"])["similarity"].count())

        country_dem_idx_class_for_present = defaultdict(list)
        country_dem_idx_class_sim_for_present = defaultdict(list)
        country_dem_idx_class_95interval_error_for_present = defaultdict(list)
        country_dem_idx_class_count_for_present = defaultdict(list)
        for key_tuple in country_pairs_group_by_dem_idx_mean:
            if country_pairs_group_by_dem_idx_count[key_tuple] >= min_pair_num:
                country_dem_idx_class_for_present[key_tuple[1]].append(key_tuple[0])
                # country_dem_idx_class_sim_for_present[key_tuple[1]].append(trans_dem_idx_sim(country_pairs_group_by_dem_idx_mean[key_tuple]))
                country_dem_idx_class_sim_for_present[key_tuple[1]].append(country_pairs_group_by_dem_idx_mean[key_tuple])
                country_dem_idx_class_count_for_present[key_tuple[1]].append(np.log(country_pairs_group_by_dem_idx_count[key_tuple]/indexes_stats_dem_idx_country_dict[key_tuple]))
        fig = plt.figure(figsize=(10, 8))
        for country_type in ['intra-country']:
            plt.plot(country_dem_idx_class_for_present[country_type],
                     country_dem_idx_class_sim_for_present[country_type], marker="o")

        # plt.ylim(0, 20)
        plt.xticks(size=20)
        plt.yticks(size=20)
        plt.xlabel('Democracy Index', fontsize=32)
        plt.ylabel('Similarity', fontsize=32)
        # plt.show()
        # actually it should just be named as dem_idx_sim.png
        plt.savefig(f"{output_dir}/dem_idx_sim_intra_country.png")

        fig = plt.figure(figsize=(10, 8))
        for country_type in ['intra-country']:
            plt.plot(country_dem_idx_class_for_present[country_type], country_dem_idx_class_count_for_present[country_type], marker="o")

        # plt.ylim(0, 20)
        plt.xticks(size=20)
        plt.yticks(size=20)
        plt.xlabel('Democracy Index', fontsize=32)
        plt.ylabel('Fraction', fontsize=32)
        # plt.show()
        # actually it should just be named as dem_idx_sim.png
        plt.savefig(f"{output_dir}/dem_idx_frac_intra_country.png")

        # plot intra-country similarity of each politcal system with democracy index
        min_pair_num = 1000

        country_pairs_group_by_dem_unitary_idx_mean = dict(country_dem_idx_pairs.groupby(["democracy_index_class", "unitary_type1"])["similarity"].mean())
        country_pairs_group_by_dem_ref_idx_mean = dict(country_dem_idx_pairs.groupby(["democracy_index_class"])["similarity"].mean())
        country_pairs_group_by_dem_unitary_idx_count = dict(country_dem_idx_pairs.groupby(["democracy_index_class", "unitary_type1"])["similarity"].count())
        country_pairs_group_by_dem_ref_idx_count = dict(country_dem_idx_pairs.groupby(["democracy_index_class"])["similarity"].count())

        country_pairs_group_by_dem_unitary_idx_list = dict(country_dem_idx_pairs.groupby(["democracy_index_class", "unitary_type1"])["similarity"].apply(list))
        country_pairs_group_by_dem_unitary_idx_95interval_error = {}
        for key_tuple in country_pairs_group_by_dem_unitary_idx_list:
            temp_low, temp_high = st.t.interval(alpha=0.95, df=len(country_pairs_group_by_dem_unitary_idx_list[key_tuple]) - 1,
                                                         loc=np.mean(country_pairs_group_by_dem_unitary_idx_list[key_tuple]),
                                                         scale=st.stats.sem(country_pairs_group_by_dem_unitary_idx_list[key_tuple]))
            country_pairs_group_by_dem_unitary_idx_95interval_error[key_tuple] = country_pairs_group_by_dem_unitary_idx_mean[key_tuple] - temp_low


        country_dem_unitary_idx_class_for_present = defaultdict(list)
        country_dem_unitary_idx_class_sim_for_present = defaultdict(list)
        country_dem_unitary_idx_class_count_for_present = defaultdict(list)
        country_dem_unitary_idx_class_95interval_error_for_present = defaultdict(list)
        for key_tuple in country_pairs_group_by_dem_unitary_idx_mean:
            if country_pairs_group_by_dem_unitary_idx_count[key_tuple] >= min_pair_num:
                country_dem_unitary_idx_class_for_present[key_tuple[1]].append(key_tuple[0])
                # country_dem_idx_class_sim_for_present[key_tuple[1]].append(trans_dem_idx_sim(country_pairs_group_by_dem_unitary_idx_mean[key_tuple]))
                country_dem_unitary_idx_class_sim_for_present[key_tuple[1]].append(country_pairs_group_by_dem_unitary_idx_mean[key_tuple])
                country_dem_unitary_idx_class_count_for_present[key_tuple[1]].append(np.log(country_pairs_group_by_dem_unitary_idx_count[key_tuple]/indexes_stats_dem_idx_country_dict[key_tuple]))
                country_dem_unitary_idx_class_95interval_error_for_present[key_tuple[1]].append(country_pairs_group_by_dem_unitary_idx_95interval_error[key_tuple])

        country_dem_ref_idx_class_for_present = []
        country_dem_ref_idx_class_sim_for_present = []
        country_dem_ref_idx_class_count_for_present = []
        for key_tuple in country_pairs_group_by_dem_ref_idx_mean:
            if country_pairs_group_by_dem_ref_idx_count[key_tuple] >= min_pair_num:
                country_dem_ref_idx_class_for_present.append(key_tuple)
                # country_dem_idx_class_sim_for_present[key_tuple[1]].append(trans_dem_idx_sim(country_pairs_group_by_dem_unitary_idx_mean[key_tuple]))
                country_dem_ref_idx_class_sim_for_present.append(country_pairs_group_by_dem_ref_idx_mean[key_tuple])
                country_dem_ref_idx_class_count_for_present.append(np.log(country_pairs_group_by_dem_ref_idx_count[key_tuple]/indexes_stats_dem_idx_country_dict[(key_tuple,'intra-country')]))

        # compute coefficient and p-value
        country_pairs_group_by_unitary_idx_coef_list = dict(country_dem_idx_pairs.groupby(["unitary_type1"])["democracy_index_class"].apply(list))
        country_pairs_group_by_unitary_sim_coef_list = dict(country_dem_idx_pairs.groupby(["unitary_type1"])["similarity"].apply(list))

        transfered_unitary_type_dict = {"unitary_republic":1, "federalism":0}
        country_dem_idx_pairs["transfered_unitary_type"] = country_dem_idx_pairs.apply(lambda x: transfered_unitary_type_dict[x["unitary_type1"]] if x["unitary_type1"] in transfered_unitary_type_dict else -1, axis=1)
        country_dem_idx_pairs = country_dem_idx_pairs[country_dem_idx_pairs["transfered_unitary_type"] != -1]
        country_pairs_unitary_coef_list = country_dem_idx_pairs["transfered_unitary_type"].to_list()
        country_pairs_sim_coef_list = country_dem_idx_pairs["similarity"].to_list()

        for unitary_type in country_alpha3_dem_unitary_dict:
            if unitary_type == "unitary_monarchies":
                continue
            print(f"{unitary_type} pearson correlation and p-value are", st.pearsonr(country_pairs_group_by_unitary_idx_coef_list[unitary_type], country_pairs_group_by_unitary_sim_coef_list[unitary_type]))
            print(f"{unitary_type} spearman correlation and p-value are", st.spearmanr(country_pairs_group_by_unitary_idx_coef_list[unitary_type], country_pairs_group_by_unitary_sim_coef_list[unitary_type]))
        print()
        print("cross unitary type pearson correlation and p-value are:", st.pearsonr(country_pairs_unitary_coef_list, country_pairs_sim_coef_list))
        print("cross unitary type spearman correlation and p-value are:", st.spearmanr(country_pairs_unitary_coef_list, country_pairs_sim_coef_list))

        fig = plt.figure(figsize=(10, 8))
        for unitary_type in country_alpha3_dem_unitary_dict:
            if unitary_type == "unitary_monarchies":
                continue
            else:
                country_dem_unitary_idx_class_for_present[unitary_type] = country_dem_unitary_idx_class_for_present[unitary_type][1:]
                country_dem_unitary_idx_class_sim_for_present[unitary_type] = country_dem_unitary_idx_class_sim_for_present[unitary_type][1:]
                country_dem_unitary_idx_class_95interval_error_for_present[unitary_type] = country_dem_unitary_idx_class_95interval_error_for_present[unitary_type][1:]
            plt.plot(country_dem_unitary_idx_class_for_present[unitary_type], country_dem_unitary_idx_class_sim_for_present[unitary_type], color=unitary_color[unitary_type], marker="o")
            plt.errorbar(country_dem_unitary_idx_class_for_present[unitary_type], country_dem_unitary_idx_class_sim_for_present[unitary_type], color=unitary_color[unitary_type], yerr=[country_dem_unitary_idx_class_95interval_error_for_present[unitary_type], country_dem_unitary_idx_class_95interval_error_for_present[unitary_type]], fmt='o', markersize=8, capsize=20)
        # plt.plot(country_dem_ref_idx_class_for_present, country_dem_ref_idx_class_sim_for_present, marker="o",color='k')

        # plt.ylim(0, 20)
        plt.xticks(size=20)
        plt.yticks(size=20)
        plt.xlabel('Democracy Index', fontsize=32)
        plt.ylabel('Similarity', fontsize=32)
        # plt.legend(["unitary republic", "unitary monarchies", "federalism"], fontsize=20)
        # plt.legend(["unitary republic", "federalism", "world"], fontsize=20)
        plt.legend(["unitary republic", "federalism"], fontsize=20)
        # plt.show()
        # actually it should just be named as dem_idx_sim.png
        plt.savefig(f"{output_dir}/dem_idx_sim_unitary_intra_country.png")


        # plot the counts of filtered-pairs

        fig = plt.figure(figsize=(10, 8))
        for unitary_type in country_alpha3_dem_unitary_dict:
            if unitary_type == "unitary_monarchies":
                continue
            if unitary_type == "federalism":
                # this process has been done when plotting the former figure
                # country_dem_unitary_idx_class_for_present[unitary_type] = country_dem_unitary_idx_class_for_present[unitary_type][1:]
                country_dem_unitary_idx_class_count_for_present[unitary_type] = country_dem_unitary_idx_class_count_for_present[unitary_type][1:]
            plt.plot(country_dem_unitary_idx_class_for_present[unitary_type], country_dem_unitary_idx_class_count_for_present[unitary_type], marker="o")
        plt.plot(country_dem_ref_idx_class_for_present, country_dem_ref_idx_class_count_for_present, marker="o", color='k')

        # plt.ylim(0, 20)
        plt.xticks(size=20)
        plt.yticks(size=20)
        plt.xlabel('Democracy Index', fontsize=32)
        plt.ylabel('Fraction', fontsize=32)
        # plt.legend(["unitary republic", "unitary monarchies", "federalism"], fontsize=20)
        plt.legend(["unitary republic", "federalism", "world"], fontsize=20)
        # plt.show()
        # actually it should just be named as dem_idx_sim.png
        plt.savefig(f"{output_dir}/dem_idx_frac_unitary_intra_country.png")



        # plot inter-country similarity of countries with different democracy index
        country_dem_idx_pairs = country_pairs.loc[country_pairs["main_country1"] != country_pairs["main_country2"]]

        # country_dem_idx_pairs["democracy_index_class"] = country_dem_idx_pairs.apply(lambda x: math.ceil(x["democracy_index1"]/2) * 2, axis=1)
        # country_dem_idx_pairs["democracy_index_class"] = country_dem_idx_pairs.apply(lambda x: math.ceil(x["democracy_index1"]), axis=1)
        # country_dem_idx_pairs["democracy_index_class"] = country_dem_idx_pairs.apply(lambda x: math.ceil(x["democracy_index1"] * 2) / 2, axis=1)


        # inter-country democracy analysis
        country_dem_idx_pairs["democracy_index_class1"] = country_dem_idx_pairs.apply(lambda x: math.ceil(x["democracy_index1"] * 2.5) / 2.5, axis=1)
        country_dem_idx_pairs["democracy_index_class2"] = country_dem_idx_pairs.apply(lambda x: math.ceil(x["democracy_index2"] * 2.5) / 2.5, axis=1)
        country_dem_idx_pairs["democracy_index_diff"] = country_dem_idx_pairs.apply(lambda x: abs(x["democracy_index_class1"] - x["democracy_index_class2"]), axis=1)

        country_dem_idx_pairs = country_dem_idx_pairs[country_dem_idx_pairs["democracy_index_diff"] < 0.6]


        # plot overall inter-country similarity with democracy index
        min_pair_num = 1000

        country_pairs_group_by_dem_idx_mean = dict(country_dem_idx_pairs.groupby(["democracy_index_class1", "country_type"])["similarity"].mean())
        country_pairs_group_by_dem_idx_count = dict(country_dem_idx_pairs.groupby(["democracy_index_class1", "country_type"])["similarity"].count())

        country_pairs_group_by_dem_idx_list = dict(country_dem_idx_pairs.groupby(["democracy_index_class1", "country_type"])["similarity"].apply(list))
        country_pairs_group_by_dem_idx_95interval_error = {}
        for key_tuple in country_pairs_group_by_dem_idx_list:
            temp_low, temp_high = st.t.interval(alpha=0.95, df=len(country_pairs_group_by_dem_idx_list[key_tuple]) - 1,
                                                         loc=np.mean(country_pairs_group_by_dem_idx_list[key_tuple]),
                                                         scale=st.stats.sem(country_pairs_group_by_dem_idx_list[key_tuple]))
            country_pairs_group_by_dem_idx_95interval_error[key_tuple] = country_pairs_group_by_dem_idx_mean[key_tuple] - temp_low

        country_dem_idx_class_for_present = defaultdict(list)
        country_dem_idx_class_sim_for_present = defaultdict(list)
        country_dem_idx_class_count_for_present = defaultdict(list)
        country_dem_idx_class_95interval_error_for_present = defaultdict(list)


        for key_tuple in country_pairs_group_by_dem_idx_mean:
            if country_pairs_group_by_dem_idx_count[key_tuple] >= min_pair_num:
                country_dem_idx_class_for_present[key_tuple[1]].append(key_tuple[0])
                # country_dem_idx_class_sim_for_present[key_tuple[1]].append(trans_dem_idx_sim(country_pairs_group_by_dem_idx_mean[key_tuple]))
                country_dem_idx_class_sim_for_present[key_tuple[1]].append(country_pairs_group_by_dem_idx_mean[key_tuple])
                country_dem_idx_class_count_for_present[key_tuple[1]].append(np.log(country_pairs_group_by_dem_idx_count[key_tuple]/indexes_stats_dem_idx_country_dict[key_tuple]))
                country_dem_idx_class_95interval_error_for_present[key_tuple[1]].append(country_pairs_group_by_dem_idx_95interval_error[key_tuple])

        # compute coefficient and p-value
        country_pairs_group_by_idx_coef_list = country_dem_idx_pairs["democracy_index_class1"].to_list()
        country_pairs_group_by_sim_coef_list = country_dem_idx_pairs["similarity"].to_list()

        print("inter-country pairs dem_idx->dissim pearson correlation and p-value are:", st.pearsonr(country_pairs_group_by_idx_coef_list, country_pairs_group_by_sim_coef_list))
        print("inter-country pairs dem_idx->dissim spearman correlation and p-value are:", st.spearmanr(country_pairs_group_by_idx_coef_list, country_pairs_group_by_sim_coef_list))

        # plot
        fig = plt.figure(figsize=(10, 8))
        for country_type in ['inter-country']:
            plt.plot(country_dem_idx_class_for_present[country_type][1:-1],country_dem_idx_class_sim_for_present[country_type][1:-1], color="blue", marker="o")
            plt.errorbar(country_dem_idx_class_for_present[country_type][1:-1],country_dem_idx_class_sim_for_present[country_type][1:-1], color="blue",
                         yerr=[country_dem_idx_class_95interval_error_for_present[country_type][1:-1],
                               country_dem_idx_class_95interval_error_for_present[country_type][1:-1]], fmt='o',
                         markersize=8, capsize=20)

        # plt.ylim(0, 20)
        plt.xticks(size=20)
        plt.yticks(size=20)
        plt.xlabel('Democracy Index', fontsize=32)
        plt.ylabel('Similarity', fontsize=32)
        # plt.show()
        # actually it should just be named as dem_idx_sim.png
        plt.savefig(f"{output_dir}/dem_idx_sim_inter_country.png")

        fig = plt.figure(figsize=(10, 8))
        for country_type in ['inter-country']:
            plt.plot(country_dem_idx_class_for_present[country_type][1:-1],country_dem_idx_class_count_for_present[country_type][1:-1], marker="o")

        # plt.ylim(0, 20)
        plt.xticks(size=20)
        plt.yticks(size=20)
        plt.xlabel('Democracy Index', fontsize=32)
        plt.ylabel('Fraction', fontsize=32)
        # plt.show()
        # actually it should just be named as dem_idx_sim.png
        plt.savefig(f"{output_dir}/dem_idx_frac_inter_country.png")

        # plot inter-country similarity of each politcal system with democracy index
        country_dem_idx_pairs["unitary_type1"] = country_dem_idx_pairs.apply(
            lambda x: find_unitary_type(x["main_country1"], country_alpha3_dem_unitary_dict), axis=1)
        country_dem_idx_pairs["unitary_type2"] = country_dem_idx_pairs.apply(
            lambda x: find_unitary_type(x["main_country2"], country_alpha3_dem_unitary_dict), axis=1)
        country_dem_idx_pairs = country_dem_idx_pairs[
            country_dem_idx_pairs["unitary_type1"] == country_dem_idx_pairs["unitary_type2"]]

        min_pair_num = 1000

        country_pairs_group_by_dem_unitary_idx_mean = dict(country_dem_idx_pairs.groupby(["democracy_index_class1", "unitary_type1"])["similarity"].mean())
        country_pairs_group_by_dem_ref_idx_mean = dict(country_dem_idx_pairs.groupby(["democracy_index_class1"])["similarity"].mean())
        country_pairs_group_by_dem_unitary_idx_count = dict(country_dem_idx_pairs.groupby(["democracy_index_class1", "unitary_type1"])["similarity"].count())
        country_pairs_group_by_dem_ref_idx_count = dict(country_dem_idx_pairs.groupby(["democracy_index_class1"])["similarity"].count())


        country_dem_unitary_idx_class_for_present = defaultdict(list)
        country_dem_unitary_idx_class_sim_for_present = defaultdict(list)
        country_dem_unitary_idx_class_count_for_present = defaultdict(list)
        for key_tuple in country_pairs_group_by_dem_unitary_idx_mean:
            if country_pairs_group_by_dem_unitary_idx_count[key_tuple] >= min_pair_num:
                country_dem_unitary_idx_class_for_present[key_tuple[1]].append(key_tuple[0])
                # country_dem_idx_class_sim_for_present[key_tuple[1]].append(trans_dem_idx_sim(country_pairs_group_by_dem_unitary_idx_mean[key_tuple]))
                country_dem_unitary_idx_class_sim_for_present[key_tuple[1]].append(country_pairs_group_by_dem_unitary_idx_mean[key_tuple])
                country_dem_unitary_idx_class_count_for_present[key_tuple[1]].append(np.log(country_pairs_group_by_dem_unitary_idx_count[key_tuple]/indexes_stats_dem_idx_country_dict[key_tuple]))

        country_dem_ref_idx_class_for_present = []
        country_dem_ref_idx_class_sim_for_present = []
        country_dem_ref_idx_class_count_for_present = []
        for key_tuple in country_pairs_group_by_dem_ref_idx_mean:
            if country_pairs_group_by_dem_ref_idx_count[key_tuple] >= min_pair_num:
                country_dem_ref_idx_class_for_present.append(key_tuple)
                # country_dem_idx_class_sim_for_present[key_tuple[1]].append(trans_dem_idx_sim(country_pairs_group_by_dem_unitary_idx_mean[key_tuple]))
                country_dem_ref_idx_class_sim_for_present.append(country_pairs_group_by_dem_ref_idx_mean[key_tuple])
                country_dem_ref_idx_class_count_for_present.append(np.log(country_pairs_group_by_dem_ref_idx_count[key_tuple]/indexes_stats_dem_idx_country_dict[(key_tuple,"inter-country")]))

        fig = plt.figure(figsize=(10, 8))
        for unitary_type in country_alpha3_dem_unitary_dict:
            plt.plot(country_dem_unitary_idx_class_for_present[unitary_type], country_dem_unitary_idx_class_sim_for_present[unitary_type], marker="o")
        plt.plot(country_dem_ref_idx_class_for_present, country_dem_ref_idx_class_sim_for_present, marker="o", color='k')


        # plt.ylim(0, 20)
        plt.xticks(size=20)
        plt.yticks(size=20)
        plt.xlabel('Democracy Index', fontsize=32)
        plt.ylabel('Similarity', fontsize=32)
        plt.legend(["unitary republic", "unitary monarchies", "federalism", "world"], fontsize=20)
        # plt.show()
        # actually it should just be named as dem_idx_sim.png
        plt.savefig(f"{output_dir}/dem_idx_sim_unitary_inter_country.png")


        fig = plt.figure(figsize=(10, 8))
        for unitary_type in country_alpha3_dem_unitary_dict:
            plt.plot(country_dem_unitary_idx_class_for_present[unitary_type], country_dem_unitary_idx_class_count_for_present[unitary_type], marker="o")
        plt.plot(country_dem_ref_idx_class_for_present, country_dem_ref_idx_class_count_for_present, marker="o", color='k')

        # plt.ylim(0, 20)
        plt.xticks(size=20)
        plt.yticks(size=20)
        plt.xlabel('Democracy Index', fontsize=32)
        plt.ylabel('Fraction', fontsize=32)
        plt.legend(["unitary republic", "unitary monarchies", "federalism", "world"], fontsize=20)
        # plt.show()
        # actually it should just be named as dem_idx_sim.png
        plt.savefig(f"{output_dir}/dem_idx_frac_unitary_inter_country.png")


    # press freedom analysis
    if args.option == "pfa":
        press_freedom_class_value = 5
        press_freedom_threshold = 10
        '''load press freedom index info'''
        country_press_freedom_list = pd.read_csv("country_info/country_press_freedom_index.csv")
        country_press_freedom_country = country_press_freedom_list["Country"].to_list()
        country_press_freedom_press_freedom = country_press_freedom_list["Press Freedom"].to_list()
        country_country_press_freedom_dict = {country_press_freedom_country[i]: country_press_freedom_press_freedom[i]  for i in range(len(country_press_freedom_country))}


        country_unitary_state_list = pd.read_csv("country_info/country_unitary_state.csv")

        country_alpha3_full_name_list = pd.merge(country_press_freedom_list, country_geography_list, on='Country')
        country_alpha3_full_name_list = pd.merge(country_alpha3_full_name_list, country_unitary_state_list, on='Country')

        country_alpha3_full_name_unitary_dict = {}
        country_alpha3_full_name_unitary_dict["unitary_republic"] = country_alpha3_full_name_list[country_alpha3_full_name_list['Unitary'] == 'unitary republics']
        country_alpha3_full_name_unitary_dict["unitary_monarchies"] = country_alpha3_full_name_list[country_alpha3_full_name_list['Unitary'] == 'unitary monarchies']
        country_alpha3_full_name_unitary_dict["federalism"] = country_alpha3_full_name_list[country_alpha3_full_name_list['Unitary'] == 'federalism']

        country_alpha3_dem_dict = { country[4]: country[2] for country in country_alpha3_full_name_list.itertuples()}
        country_alpha3_dem_unitary_dict = {}
        country_alpha3_dem_unitary_dict["unitary_republic"] = { country[4]: country[2] for country in country_alpha3_full_name_unitary_dict["unitary_republic"].itertuples()}
        country_alpha3_dem_unitary_dict["unitary_monarchies"] = { country[4]: country[2] for country in country_alpha3_full_name_unitary_dict["unitary_monarchies"].itertuples()}
        country_alpha3_dem_unitary_dict["federalism"] = { country[4]: country[2] for country in country_alpha3_full_name_unitary_dict["federalism"].itertuples()}

        # first draw a scatter plot for similarity of each country, x is the democracy index
        country_pairs = pairs.copy(deep=True)
        country_pairs["press_freedom1"] = country_pairs.apply(lambda x: country_country_press_freedom_dict[x["country_full_name1"]] if x["country_full_name1"] in country_country_press_freedom_dict else np.nan, axis=1)
        country_pairs["press_freedom2"] = country_pairs.apply(lambda x: country_country_press_freedom_dict[x["country_full_name2"]] if x["country_full_name2"] in country_country_press_freedom_dict else np.nan, axis=1)

        country_pairs = country_pairs.dropna(subset=["main_country1", "main_country2", "press_freedom1", "press_freedom2"])
        country_pairs['country_type'] = country_pairs.apply(lambda x: "intra-country" if (x["main_country1"] == x["main_country2"]) else "inter-country", axis=1)
        country_sim_dict = defaultdict(list)
        # country_pairs.apply(lambda x: country_sim_dict[x["main_country1"]].append(trans_node_sim(x["similarity"])) if x["main_country1"] == x["main_country2"] else 0, axis=1)
        country_pairs.apply(lambda x: country_sim_dict[x["main_country1"]].append(x["similarity"]) if x["main_country1"] == x["main_country2"] else 0, axis=1)

        # loading index data for statistics
        indexes_stats_dem_idx = indexes_stats.drop(["relatime"], axis=1)
        indexes_stats_dem_idx["press_freedom"] = indexes_stats_dem_idx.apply(lambda x: country_country_press_freedom_dict[x["country_full_name"]] if x["country_full_name"] in country_country_press_freedom_dict else np.nan, axis=1)

        indexes_stats_dem_idx = indexes_stats_dem_idx.dropna(subset=["main_country", "press_freedom"])

        indexes_stats_dem_idx["press_freedom_class"] = indexes_stats_dem_idx.apply(lambda x: math.ceil(x["press_freedom"] / press_freedom_class_value) * press_freedom_class_value, axis=1)
        indexes_stats_dem_idx["unitary_type"] = indexes_stats_dem_idx.apply(lambda x: find_unitary_type(x["main_country"], country_alpha3_dem_unitary_dict), axis=1)
        indexes_stats_group_by_dem_idx_count = dict(indexes_stats_dem_idx.groupby(["main_country"])["count"].sum())

        indexes_stats_dem_idx["dem_count"] = indexes_stats_dem_idx.apply(lambda x: indexes_stats_group_by_dem_idx_count[x["main_country"]], axis=1)
        indexes_stats_dem_idx = indexes_stats_dem_idx.drop(["lang", "media_id", "count", "pub_country", "about_country"], axis=1)
        indexes_stats_dem_idx = indexes_stats_dem_idx.drop_duplicates()

        indexes_stats_dem_idx_country = pd.DataFrame()
        indexes_stats_dem_idx_unitary_type = pd.DataFrame()
        indexes_stats_dem_idx_country_dict = {}
        indexes_stats_dem_idx_country["press_freedom_class"] = indexes_stats_dem_idx.groupby('press_freedom_class').agg({'dem_count': list}).reset_index().apply(lambda d: d.press_freedom_class, axis=1)
        indexes_stats_dem_idx_country["intra_country"] = indexes_stats_dem_idx.groupby('press_freedom_class').agg({'dem_count': list}).reset_index().apply(lambda d: dem_class_count(d['dem_count'], "intra_country"), axis=1)
        indexes_stats_dem_idx_country["simple_intra_country"] = indexes_stats_dem_idx.groupby('press_freedom_class').agg({'dem_count': list}).reset_index().apply(lambda d: dem_class_count(d['dem_count'], "intra_country"), axis=1)
        a = indexes_stats_dem_idx.groupby(['press_freedom_class', 'unitary_type'])
        b = a.agg({'dem_count': list})
        indexes_stats_dem_idx_unitary_type["press_freedom_class"] = indexes_stats_dem_idx.groupby(['press_freedom_class', 'unitary_type']).agg({'dem_count': list}).reset_index().apply(lambda d: d.press_freedom_class, axis=1)
        indexes_stats_dem_idx_unitary_type["unitary_type"] = indexes_stats_dem_idx.groupby(['press_freedom_class', 'unitary_type']).agg({'dem_count': list}).reset_index().apply(lambda d: d.unitary_type, axis=1)
        indexes_stats_dem_idx_unitary_type["intra_country"] = indexes_stats_dem_idx.groupby(['press_freedom_class', 'unitary_type']).agg({'dem_count': list}).reset_index().apply(lambda d: dem_class_count(d['dem_count'], "intra_country"), axis=1)
        for dem_idx_country_count in indexes_stats_dem_idx_country.itertuples():
            cur_press_freedom_class = dem_idx_country_count[1]
            cur_intra_country_count = dem_idx_country_count[2]
            cur_inter_country_count = dem_idx_country_count[3]

            indexes_stats_dem_idx_country_dict[(cur_press_freedom_class, "intra-country")] = cur_intra_country_count
            indexes_stats_dem_idx_country_dict[(cur_press_freedom_class, "inter-country")] = cur_inter_country_count

        for dem_idx_unitary_count in indexes_stats_dem_idx_unitary_type.itertuples():
            cur_press_freedom_class = dem_idx_unitary_count[1]
            cur_unitary_type = dem_idx_unitary_count[2]
            cur_intra_unitary_count = dem_idx_unitary_count[3]

            indexes_stats_dem_idx_country_dict[(cur_press_freedom_class, cur_unitary_type)] = cur_intra_unitary_count
        # for test

        min_pair_num = 50

        country_dem_for_present = []
        country_sim_for_present = []
        country_dem_unitary_for_present = defaultdict(list)
        country_sim_unitary_for_present = defaultdict(list)

        for country in country_sim_dict:
            if len(country_sim_dict[country]) >= min_pair_num:
                try:
                    country_dem_for_present.append(country_alpha3_dem_dict[country])
                    country_sim_for_present.append(np.mean(country_sim_dict[country]))
                except:
                    pass
                try:
                    for unitary_type in country_alpha3_dem_unitary_dict:
                        if country in country_alpha3_dem_unitary_dict[unitary_type]:
                            country_dem_unitary_for_present[unitary_type].append(country_alpha3_dem_dict[country])
                            country_sim_unitary_for_present[unitary_type].append(np.mean(country_sim_dict[country]))
                except:
                    pass

        dem_sim_scatter_dict = {"Press Freedom Index":country_dem_for_present, "Similarity":country_sim_for_present}
        dem_sim_scatter_df = pd.DataFrame(dem_sim_scatter_dict)
        fig = sns.lmplot(x="Press Freedom Index", y="Similarity", data=dem_sim_scatter_df, order=3, ci=95)
        fig.savefig(f"{output_dir}/pf_sim_scatter.png")

        # plot scatter according to their political system type
        unitary_color = {"unitary_republic":"red", "unitary_monarchies":"blue", "federalism":"green"}
        fig = plt.figure(figsize=(10, 8))
        for unitary_type in country_alpha3_dem_unitary_dict:
            plt.scatter(country_dem_unitary_for_present[unitary_type], country_sim_unitary_for_present[unitary_type], c= unitary_color[unitary_type])
        plt.xticks(size=20)
        plt.yticks(size=20)
        plt.xlabel('Press Freedom Index', fontsize=32)
        plt.ylabel('Similarity', fontsize=32)
        plt.legend(["unitary republic", "unitary monarchies", "federalism"],fontsize= 20)
        # plt.show()
        plt.savefig(f"{output_dir}/pf_sim_unitary_scatter.png")

        # plot intra-country similarity of countries with different Press Freedom Index
        country_dem_idx_pairs = country_pairs.loc[country_pairs["main_country1"] == country_pairs["main_country2"]]

        # country_dem_idx_pairs["press_freedom_class"] = country_dem_idx_pairs.apply(lambda x: math.ceil(x["press_freedom1"]/2) * 2, axis=1)
        # country_dem_idx_pairs["press_freedom_class"] = country_dem_idx_pairs.apply(lambda x: math.ceil(x["press_freedom1"]), axis=1)
        # country_dem_idx_pairs["press_freedom_class"] = country_dem_idx_pairs.apply(lambda x: math.ceil(x["press_freedom1"] * 2) / 2, axis=1)
        country_dem_idx_pairs["press_freedom_class"] = country_dem_idx_pairs.apply(lambda x: math.ceil(x["press_freedom1"] / press_freedom_class_value) * press_freedom_class_value, axis=1)
        country_dem_idx_pairs["unitary_type1"] = country_dem_idx_pairs.apply(lambda x: find_unitary_type(x["main_country1"], country_alpha3_dem_unitary_dict), axis=1)

        # plot overall intra-country similarity with Press Freedom Index
        min_pair_num = 1000

        country_pairs_group_by_dem_idx_mean = dict(country_dem_idx_pairs.groupby(["press_freedom_class", "country_type"])["similarity"].mean())
        country_pairs_group_by_dem_idx_count = dict(country_dem_idx_pairs.groupby(["press_freedom_class", "country_type"])["similarity"].count())

        country_dem_idx_class_for_present = defaultdict(list)
        country_dem_idx_class_sim_for_present = defaultdict(list)
        country_dem_idx_class_95interval_error_for_present = defaultdict(list)
        country_dem_idx_class_count_for_present = defaultdict(list)
        for key_tuple in country_pairs_group_by_dem_idx_mean:
            if country_pairs_group_by_dem_idx_count[key_tuple] >= min_pair_num:
                country_dem_idx_class_for_present[key_tuple[1]].append(key_tuple[0])
                # country_dem_idx_class_sim_for_present[key_tuple[1]].append(trans_dem_idx_sim(country_pairs_group_by_dem_idx_mean[key_tuple]))
                country_dem_idx_class_sim_for_present[key_tuple[1]].append(country_pairs_group_by_dem_idx_mean[key_tuple])
                country_dem_idx_class_count_for_present[key_tuple[1]].append(np.log(country_pairs_group_by_dem_idx_count[key_tuple]/indexes_stats_dem_idx_country_dict[key_tuple]))
        fig = plt.figure(figsize=(10, 8))
        for country_type in ['intra-country']:
            plt.plot(country_dem_idx_class_for_present[country_type],
                     country_dem_idx_class_sim_for_present[country_type], marker="o")

        # plt.ylim(0, 20)
        plt.xticks(size=20)
        plt.yticks(size=20)
        plt.xlabel('Press Freedom Index', fontsize=32)
        plt.ylabel('Similarity', fontsize=32)
        # plt.show()
        # actually it should just be named as dem_idx_sim.png
        plt.savefig(f"{output_dir}/pf_idx_sim_intra_country.png")

        fig = plt.figure(figsize=(10, 8))
        for country_type in ['intra-country']:
            plt.plot(country_dem_idx_class_for_present[country_type], country_dem_idx_class_count_for_present[country_type], marker="o")

        # plt.ylim(0, 20)
        plt.xticks(size=20)
        plt.yticks(size=20)
        plt.xlabel('Press Freedom Index', fontsize=32)
        plt.ylabel('Fraction', fontsize=32)
        # plt.show()
        # actually it should just be named as dem_idx_sim.png
        plt.savefig(f"{output_dir}/pf_idx_frac_intra_country.png")

        # plot intra-country similarity of each politcal system with Press Freedom Index
        min_pair_num = 1000

        country_pairs_group_by_dem_unitary_idx_mean = dict(country_dem_idx_pairs.groupby(["press_freedom_class", "unitary_type1"])["similarity"].mean())
        country_pairs_group_by_dem_ref_idx_mean = dict(country_dem_idx_pairs.groupby(["press_freedom_class"])["similarity"].mean())
        country_pairs_group_by_dem_unitary_idx_count = dict(country_dem_idx_pairs.groupby(["press_freedom_class", "unitary_type1"])["similarity"].count())
        country_pairs_group_by_dem_ref_idx_count = dict(country_dem_idx_pairs.groupby(["press_freedom_class"])["similarity"].count())

        country_pairs_group_by_dem_unitary_idx_list = dict(country_dem_idx_pairs.groupby(["press_freedom_class", "unitary_type1"])["similarity"].apply(list))
        country_pairs_group_by_dem_unitary_idx_95interval_error = {}
        for key_tuple in country_pairs_group_by_dem_unitary_idx_list:
            temp_low, temp_high = st.t.interval(alpha=0.95, df=len(country_pairs_group_by_dem_unitary_idx_list[key_tuple]) - 1,
                                                         loc=np.mean(country_pairs_group_by_dem_unitary_idx_list[key_tuple]),
                                                         scale=st.stats.sem(country_pairs_group_by_dem_unitary_idx_list[key_tuple]))
            country_pairs_group_by_dem_unitary_idx_95interval_error[key_tuple] = country_pairs_group_by_dem_unitary_idx_mean[key_tuple] - temp_low


        country_dem_unitary_idx_class_for_present = defaultdict(list)
        country_dem_unitary_idx_class_sim_for_present = defaultdict(list)
        country_dem_unitary_idx_class_count_for_present = defaultdict(list)
        country_dem_unitary_idx_class_95interval_error_for_present = defaultdict(list)
        for key_tuple in country_pairs_group_by_dem_unitary_idx_mean:
            if country_pairs_group_by_dem_unitary_idx_count[key_tuple] >= min_pair_num:
                country_dem_unitary_idx_class_for_present[key_tuple[1]].append(key_tuple[0])
                # country_dem_idx_class_sim_for_present[key_tuple[1]].append(trans_dem_idx_sim(country_pairs_group_by_dem_unitary_idx_mean[key_tuple]))
                country_dem_unitary_idx_class_sim_for_present[key_tuple[1]].append(country_pairs_group_by_dem_unitary_idx_mean[key_tuple])
                country_dem_unitary_idx_class_count_for_present[key_tuple[1]].append(np.log(country_pairs_group_by_dem_unitary_idx_count[key_tuple]/indexes_stats_dem_idx_country_dict[key_tuple]))
                country_dem_unitary_idx_class_95interval_error_for_present[key_tuple[1]].append(country_pairs_group_by_dem_unitary_idx_95interval_error[key_tuple])

        country_dem_ref_idx_class_for_present = []
        country_dem_ref_idx_class_sim_for_present = []
        country_dem_ref_idx_class_count_for_present = []
        for key_tuple in country_pairs_group_by_dem_ref_idx_mean:
            if country_pairs_group_by_dem_ref_idx_count[key_tuple] >= min_pair_num:
                country_dem_ref_idx_class_for_present.append(key_tuple)
                # country_dem_idx_class_sim_for_present[key_tuple[1]].append(trans_dem_idx_sim(country_pairs_group_by_dem_unitary_idx_mean[key_tuple]))
                country_dem_ref_idx_class_sim_for_present.append(country_pairs_group_by_dem_ref_idx_mean[key_tuple])
                country_dem_ref_idx_class_count_for_present.append(np.log(country_pairs_group_by_dem_ref_idx_count[key_tuple]/indexes_stats_dem_idx_country_dict[(key_tuple,'intra-country')]))

        # compute coefficient and p-value
        country_pairs_group_by_unitary_idx_coef_list = dict(country_dem_idx_pairs.groupby(["unitary_type1"])["press_freedom_class"].apply(list))
        country_pairs_group_by_unitary_sim_coef_list = dict(country_dem_idx_pairs.groupby(["unitary_type1"])["similarity"].apply(list))

        transfered_unitary_type_dict = {"unitary_republic":1, "federalism":0}
        country_dem_idx_pairs["transfered_unitary_type"] = country_dem_idx_pairs.apply(lambda x: transfered_unitary_type_dict[x["unitary_type1"]] if x["unitary_type1"] in transfered_unitary_type_dict else -1, axis=1)
        country_dem_idx_pairs = country_dem_idx_pairs[country_dem_idx_pairs["transfered_unitary_type"] != -1]
        country_pairs_unitary_coef_list = country_dem_idx_pairs["transfered_unitary_type"].to_list()
        country_pairs_sim_coef_list = country_dem_idx_pairs["similarity"].to_list()

        for unitary_type in country_alpha3_dem_unitary_dict:
            if unitary_type == "unitary_monarchies":
                continue
            print(f"{unitary_type} pearson correlation and p-value are", st.pearsonr(country_pairs_group_by_unitary_idx_coef_list[unitary_type], country_pairs_group_by_unitary_sim_coef_list[unitary_type]))
            print(f"{unitary_type} spearman correlation and p-value are", st.spearmanr(country_pairs_group_by_unitary_idx_coef_list[unitary_type], country_pairs_group_by_unitary_sim_coef_list[unitary_type]))
        print()
        print("cross unitary type pearson correlation and p-value are:", st.pearsonr(country_pairs_unitary_coef_list, country_pairs_sim_coef_list))
        print("cross unitary type spearman correlation and p-value are:", st.spearmanr(country_pairs_unitary_coef_list, country_pairs_sim_coef_list))

        fig = plt.figure(figsize=(10, 8))
        for unitary_type in country_alpha3_dem_unitary_dict:
            if unitary_type == "unitary_monarchies":
                continue
            else:
                country_dem_unitary_idx_class_for_present[unitary_type] = country_dem_unitary_idx_class_for_present[unitary_type][1:]
                country_dem_unitary_idx_class_sim_for_present[unitary_type] = country_dem_unitary_idx_class_sim_for_present[unitary_type][1:]
                country_dem_unitary_idx_class_95interval_error_for_present[unitary_type] = country_dem_unitary_idx_class_95interval_error_for_present[unitary_type][1:]
            plt.plot(country_dem_unitary_idx_class_for_present[unitary_type], country_dem_unitary_idx_class_sim_for_present[unitary_type], color=unitary_color[unitary_type], marker="o")
            plt.errorbar(country_dem_unitary_idx_class_for_present[unitary_type], country_dem_unitary_idx_class_sim_for_present[unitary_type], color=unitary_color[unitary_type], yerr=[country_dem_unitary_idx_class_95interval_error_for_present[unitary_type], country_dem_unitary_idx_class_95interval_error_for_present[unitary_type]], fmt='o', markersize=8, capsize=20)
        # plt.plot(country_dem_ref_idx_class_for_present, country_dem_ref_idx_class_sim_for_present, marker="o",color='k')

        # plt.ylim(0, 20)
        plt.xticks(size=20)
        plt.yticks(size=20)
        plt.xlabel('Press Freedom Index', fontsize=32)
        plt.ylabel('Similarity', fontsize=32)
        # plt.legend(["unitary republic", "unitary monarchies", "federalism"], fontsize=20)
        # plt.legend(["unitary republic", "federalism", "world"], fontsize=20)
        plt.legend(["unitary republic", "federalism"], fontsize=20)
        # plt.show()
        # actually it should just be named as dem_idx_sim.png
        plt.savefig(f"{output_dir}/pf_idx_sim_unitary_intra_country.png")


        # plot the counts of filtered-pairs

        fig = plt.figure(figsize=(10, 8))
        for unitary_type in country_alpha3_dem_unitary_dict:
            if unitary_type == "unitary_monarchies":
                continue
            if unitary_type == "federalism":
                # this process has been done when plotting the former figure
                # country_dem_unitary_idx_class_for_present[unitary_type] = country_dem_unitary_idx_class_for_present[unitary_type][1:]
                country_dem_unitary_idx_class_count_for_present[unitary_type] = country_dem_unitary_idx_class_count_for_present[unitary_type][1:]
            plt.plot(country_dem_unitary_idx_class_for_present[unitary_type], country_dem_unitary_idx_class_count_for_present[unitary_type], marker="o")
        plt.plot(country_dem_ref_idx_class_for_present, country_dem_ref_idx_class_count_for_present, marker="o", color='k')

        # plt.ylim(0, 20)
        plt.xticks(size=20)
        plt.yticks(size=20)
        plt.xlabel('Press Freedom Index', fontsize=32)
        plt.ylabel('Fraction', fontsize=32)
        # plt.legend(["unitary republic", "unitary monarchies", "federalism"], fontsize=20)
        plt.legend(["unitary republic", "federalism", "world"], fontsize=20)
        # plt.show()
        # actually it should just be named as dem_idx_sim.png
        plt.savefig(f"{output_dir}/pf_idx_frac_unitary_intra_country.png")



        # plot inter-country similarity of countries with different Press Freedom Index
        country_dem_idx_pairs = country_pairs.loc[country_pairs["main_country1"] != country_pairs["main_country2"]]

        # country_dem_idx_pairs["press_freedom_class"] = country_dem_idx_pairs.apply(lambda x: math.ceil(x["press_freedom1"]/2) * 2, axis=1)
        # country_dem_idx_pairs["press_freedom_class"] = country_dem_idx_pairs.apply(lambda x: math.ceil(x["press_freedom1"]), axis=1)
        # country_dem_idx_pairs["press_freedom_class"] = country_dem_idx_pairs.apply(lambda x: math.ceil(x["press_freedom1"] * 2) / 2, axis=1)


        # inter-country democracy analysis
        country_dem_idx_pairs["press_freedom_class1"] = country_dem_idx_pairs.apply(lambda x: math.ceil(x["press_freedom1"] / press_freedom_class_value) * press_freedom_class_value, axis=1)
        country_dem_idx_pairs["press_freedom_class2"] = country_dem_idx_pairs.apply(lambda x: math.ceil(x["press_freedom2"] / press_freedom_class_value) * press_freedom_class_value, axis=1)
        country_dem_idx_pairs["press_freedom_diff"] = country_dem_idx_pairs.apply(lambda x: abs(x["press_freedom_class1"] - x["press_freedom_class2"]), axis=1)

        country_dem_idx_pairs = country_dem_idx_pairs[country_dem_idx_pairs["press_freedom_diff"] < press_freedom_threshold]


        # plot overall inter-country similarity with Press Freedom Index
        min_pair_num = 1000

        country_pairs_group_by_dem_idx_mean = dict(country_dem_idx_pairs.groupby(["press_freedom_class1", "country_type"])["similarity"].mean())
        country_pairs_group_by_dem_idx_count = dict(country_dem_idx_pairs.groupby(["press_freedom_class1", "country_type"])["similarity"].count())

        country_pairs_group_by_dem_idx_list = dict(country_dem_idx_pairs.groupby(["press_freedom_class1", "country_type"])["similarity"].apply(list))
        country_pairs_group_by_dem_idx_95interval_error = {}
        for key_tuple in country_pairs_group_by_dem_idx_list:
            temp_low, temp_high = st.t.interval(alpha=0.95, df=len(country_pairs_group_by_dem_idx_list[key_tuple]) - 1,
                                                         loc=np.mean(country_pairs_group_by_dem_idx_list[key_tuple]),
                                                         scale=st.stats.sem(country_pairs_group_by_dem_idx_list[key_tuple]))
            country_pairs_group_by_dem_idx_95interval_error[key_tuple] = country_pairs_group_by_dem_idx_mean[key_tuple] - temp_low

        country_dem_idx_class_for_present = defaultdict(list)
        country_dem_idx_class_sim_for_present = defaultdict(list)
        country_dem_idx_class_count_for_present = defaultdict(list)
        country_dem_idx_class_95interval_error_for_present = defaultdict(list)


        for key_tuple in country_pairs_group_by_dem_idx_mean:
            if country_pairs_group_by_dem_idx_count[key_tuple] >= min_pair_num:
                country_dem_idx_class_for_present[key_tuple[1]].append(key_tuple[0])
                # country_dem_idx_class_sim_for_present[key_tuple[1]].append(trans_dem_idx_sim(country_pairs_group_by_dem_idx_mean[key_tuple]))
                country_dem_idx_class_sim_for_present[key_tuple[1]].append(country_pairs_group_by_dem_idx_mean[key_tuple])
                country_dem_idx_class_count_for_present[key_tuple[1]].append(np.log(country_pairs_group_by_dem_idx_count[key_tuple]/indexes_stats_dem_idx_country_dict[key_tuple]))
                country_dem_idx_class_95interval_error_for_present[key_tuple[1]].append(country_pairs_group_by_dem_idx_95interval_error[key_tuple])

        # compute coefficient and p-value
        country_pairs_group_by_idx_coef_list = country_dem_idx_pairs["press_freedom_class1"].to_list()
        country_pairs_group_by_sim_coef_list = country_dem_idx_pairs["similarity"].to_list()

        print("inter-country pairs pf_idx->dissim pearson correlation and p-value are:", st.pearsonr(country_pairs_group_by_idx_coef_list, country_pairs_group_by_sim_coef_list))
        print("inter-country pairs pf_idx->dissim spearman correlation and p-value are:", st.spearmanr(country_pairs_group_by_idx_coef_list, country_pairs_group_by_sim_coef_list))

        # plot
        fig = plt.figure(figsize=(10, 8))
        for country_type in ['inter-country']:
            plt.plot(country_dem_idx_class_for_present[country_type][1:-1],country_dem_idx_class_sim_for_present[country_type][1:-1], color="blue", marker="o")
            plt.errorbar(country_dem_idx_class_for_present[country_type][1:-1],country_dem_idx_class_sim_for_present[country_type][1:-1], color="blue",
                         yerr=[country_dem_idx_class_95interval_error_for_present[country_type][1:-1],
                               country_dem_idx_class_95interval_error_for_present[country_type][1:-1]], fmt='o',
                         markersize=8, capsize=20)

        # plt.ylim(0, 20)
        plt.xticks(size=20)
        plt.yticks(size=20)
        plt.xlabel('Press Freedom Index', fontsize=32)
        plt.ylabel('Similarity', fontsize=32)
        # plt.show()
        # actually it should just be named as dem_idx_sim.png
        plt.savefig(f"{output_dir}/pf_idx_sim_inter_country.png")

        fig = plt.figure(figsize=(10, 8))
        for country_type in ['inter-country']:
            plt.plot(country_dem_idx_class_for_present[country_type][1:-1],country_dem_idx_class_count_for_present[country_type][1:-1], marker="o")

        # plt.ylim(0, 20)
        plt.xticks(size=20)
        plt.yticks(size=20)
        plt.xlabel('Press Freedom Index', fontsize=32)
        plt.ylabel('Fraction', fontsize=32)
        # plt.show()
        # actually it should just be named as dem_idx_sim.png
        plt.savefig(f"{output_dir}/pf_idx_frac_inter_country.png")

        # plot inter-country similarity of each politcal system with Press Freedom Index
        country_dem_idx_pairs["unitary_type1"] = country_dem_idx_pairs.apply(
            lambda x: find_unitary_type(x["main_country1"], country_alpha3_dem_unitary_dict), axis=1)
        country_dem_idx_pairs["unitary_type2"] = country_dem_idx_pairs.apply(
            lambda x: find_unitary_type(x["main_country2"], country_alpha3_dem_unitary_dict), axis=1)
        country_dem_idx_pairs = country_dem_idx_pairs[
            country_dem_idx_pairs["unitary_type1"] == country_dem_idx_pairs["unitary_type2"]]

        min_pair_num = 1000

        country_pairs_group_by_dem_unitary_idx_mean = dict(country_dem_idx_pairs.groupby(["press_freedom_class1", "unitary_type1"])["similarity"].mean())
        country_pairs_group_by_dem_ref_idx_mean = dict(country_dem_idx_pairs.groupby(["press_freedom_class1"])["similarity"].mean())
        country_pairs_group_by_dem_unitary_idx_count = dict(country_dem_idx_pairs.groupby(["press_freedom_class1", "unitary_type1"])["similarity"].count())
        country_pairs_group_by_dem_ref_idx_count = dict(country_dem_idx_pairs.groupby(["press_freedom_class1"])["similarity"].count())


        country_dem_unitary_idx_class_for_present = defaultdict(list)
        country_dem_unitary_idx_class_sim_for_present = defaultdict(list)
        country_dem_unitary_idx_class_count_for_present = defaultdict(list)
        for key_tuple in country_pairs_group_by_dem_unitary_idx_mean:
            if country_pairs_group_by_dem_unitary_idx_count[key_tuple] >= min_pair_num:
                country_dem_unitary_idx_class_for_present[key_tuple[1]].append(key_tuple[0])
                # country_dem_idx_class_sim_for_present[key_tuple[1]].append(trans_dem_idx_sim(country_pairs_group_by_dem_unitary_idx_mean[key_tuple]))
                country_dem_unitary_idx_class_sim_for_present[key_tuple[1]].append(country_pairs_group_by_dem_unitary_idx_mean[key_tuple])
                country_dem_unitary_idx_class_count_for_present[key_tuple[1]].append(np.log(country_pairs_group_by_dem_unitary_idx_count[key_tuple]/indexes_stats_dem_idx_country_dict[key_tuple]))

        country_dem_ref_idx_class_for_present = []
        country_dem_ref_idx_class_sim_for_present = []
        country_dem_ref_idx_class_count_for_present = []
        for key_tuple in country_pairs_group_by_dem_ref_idx_mean:
            if country_pairs_group_by_dem_ref_idx_count[key_tuple] >= min_pair_num:
                country_dem_ref_idx_class_for_present.append(key_tuple)
                # country_dem_idx_class_sim_for_present[key_tuple[1]].append(trans_dem_idx_sim(country_pairs_group_by_dem_unitary_idx_mean[key_tuple]))
                country_dem_ref_idx_class_sim_for_present.append(country_pairs_group_by_dem_ref_idx_mean[key_tuple])
                country_dem_ref_idx_class_count_for_present.append(np.log(country_pairs_group_by_dem_ref_idx_count[key_tuple]/indexes_stats_dem_idx_country_dict[(key_tuple,"inter-country")]))

        fig = plt.figure(figsize=(10, 8))
        for unitary_type in country_alpha3_dem_unitary_dict:
            plt.plot(country_dem_unitary_idx_class_for_present[unitary_type], country_dem_unitary_idx_class_sim_for_present[unitary_type], marker="o")
        plt.plot(country_dem_ref_idx_class_for_present, country_dem_ref_idx_class_sim_for_present, marker="o", color='k')


        # plt.ylim(0, 20)
        plt.xticks(size=20)
        plt.yticks(size=20)
        plt.xlabel('Press Freedom Index', fontsize=32)
        plt.ylabel('Similarity', fontsize=32)
        plt.legend(["unitary republic", "unitary monarchies", "federalism", "world"], fontsize=20)
        # plt.show()
        # actually it should just be named as dem_idx_sim.png
        plt.savefig(f"{output_dir}/pf_idx_sim_unitary_inter_country.png")


        fig = plt.figure(figsize=(10, 8))
        for unitary_type in country_alpha3_dem_unitary_dict:
            plt.plot(country_dem_unitary_idx_class_for_present[unitary_type], country_dem_unitary_idx_class_count_for_present[unitary_type], marker="o")
        plt.plot(country_dem_ref_idx_class_for_present, country_dem_ref_idx_class_count_for_present, marker="o", color='k')

        # plt.ylim(0, 20)
        plt.xticks(size=20)
        plt.yticks(size=20)
        plt.xlabel('Press Freedom Index', fontsize=32)
        plt.ylabel('Fraction', fontsize=32)
        plt.legend(["unitary republic", "unitary monarchies", "federalism", "world"], fontsize=20)
        # plt.show()
        # actually it should just be named as dem_idx_sim.png
        plt.savefig(f"{output_dir}/pf_idx_frac_unitary_inter_country.png")

    # temporal analysis,
    # haven't apply fraction computation for cross-group analysis and per-country within gorup analysis since I haven't decided the final figure form of them.
    if args.option == "ta":
        time_bin = 12

        # load political group info
        # dict for speedy query
        # one country or country pair can belong to multiple political group
        political_group_df_dict = {}
        political_group_alpha3_dict = {}
        for group in political_group_list:
            political_group_df_dict[group] = pd.read_csv(f"country_info/political_group/{group}.csv")
            # nato csv contains some other countries
            if group == "nato":
                political_group_df_dict["nato"] = political_group_df_dict["nato"][political_group_df_dict["nato"]["Category"] == "NATO"]
            political_group_alpha3_dict[group] = {alpha3: 1 for alpha3 in political_group_df_dict[group]['Alpha-3'].to_list()}


        political_group_pair = pairs.copy(deep=True)
        political_group_pair = political_group_pair.dropna(subset=["longitude1", "longitude2", "latitude1", "latitude2"])
        political_group_pair["political_groups"] = political_group_pair.apply( lambda x: match_within_political_group(x["main_country1"], x["main_country2"], political_group_list, political_group_alpha3_dict), axis=1)
        political_group_pair["political_groups_num"] = political_group_pair.apply( lambda x: len(x["political_groups"]), axis=1)
        political_group_pair["political_across_groups"] = political_group_pair.apply(lambda x: match_across_political_group(x["main_country1"], x["main_country2"], political_group_list, political_group_alpha3_dict), axis=1)
        political_group_pair["time_bin"] = political_group_pair.apply( lambda x: round((x['date1'] + x['date2'])/2/time_bin), axis=1)


        # loading index data for statistics
        indexes_stats_time = indexes_stats.dropna(subset=["main_country", "relatime"])
        indexes_stats_time["time_bin"] = indexes_stats.apply(lambda x: round(x["relatime"] / time_bin), axis=1)

        indexes_stats_group_by_time_count = dict(indexes_stats_time.groupby(["main_country", 'time_bin'])["count"].sum())
        indexes_stats_time["time_count"] = indexes_stats_time.apply(lambda x: indexes_stats_group_by_time_count[x["main_country"],x["time_bin"]], axis=1)

        indexes_stats_time = indexes_stats_time.drop(["relatime","lang", "media_id", "count", "pub_country", "about_country"], axis=1)
        indexes_stats_time = indexes_stats_time.drop_duplicates()

        indexes_stats_time["political_groups"] = indexes_stats_time.apply(lambda x: [group for group in political_group_alpha3_dict if x["main_country"] in political_group_alpha3_dict[group]], axis=1)
        indexes_stats_time["political_groups_num"] = indexes_stats_time.apply(lambda x: len(x["political_groups"]),axis=1)

        indexes_stats_time_dict = defaultdict(lambda: defaultdict(list))
        indexes_stats_time_country_dict = defaultdict(lambda: defaultdict(int))
        indexes_stats_time_group_dict = defaultdict(lambda: defaultdict(int))
        intra_country_indexes_stats_time_group_dict = defaultdict(lambda: defaultdict(int))
        inter_country_indexes_stats_time_group_dict = defaultdict(lambda: defaultdict(int))
        indexes_stats_time_group_ref_dict = defaultdict(int)
        indexes_stats_time_country_ref_dict = defaultdict(lambda: defaultdict(int))
        indexes_stats_time_country_2_groups_ref_dict = defaultdict(lambda: defaultdict(int))
        intra_country_indexes_stats_time_group_ref_dict = defaultdict(int)
        inter_country_indexes_stats_time_group_ref_dict = defaultdict(int)

        for time_political_group in indexes_stats_time.itertuples():
            cur_main_country = time_political_group[2]
            cur_time_bin = time_political_group[7]
            cur_count = time_political_group[8]
            cur_groups = time_political_group[9]

            indexes_stats_time_country_dict[cur_main_country][cur_time_bin] = cur_count * (cur_count-1)
            for cur_group in cur_groups:
                indexes_stats_time_country_ref_dict[cur_group][cur_time_bin] += indexes_stats_time_country_dict[cur_main_country][cur_time_bin]
                indexes_stats_time_dict[cur_group][cur_time_bin].append(cur_count)

            if 'nato' in cur_groups and 'eunion' in cur_groups:
                indexes_stats_time_country_2_groups_ref_dict['nato-eunion'][cur_time_bin] += indexes_stats_time_country_dict[cur_main_country][cur_time_bin]

        for cur_group in indexes_stats_time_dict:
            for cur_time_bin in indexes_stats_time_dict[cur_group]:
                # intra_country_count
                for cur_count in indexes_stats_time_dict[cur_group][cur_time_bin]:
                    intra_country_indexes_stats_time_group_dict[cur_group][cur_time_bin] += cur_count * (cur_count-1)
                # inter_country_count
                for i in range(len(indexes_stats_time_dict[cur_group][cur_time_bin])):
                    for j in range(len(indexes_stats_time_dict[cur_group][cur_time_bin])):
                        if i != j:
                            inter_country_indexes_stats_time_group_dict[cur_group][cur_time_bin] += indexes_stats_time_dict[cur_group][cur_time_bin][i] * indexes_stats_time_dict[cur_group][cur_time_bin][j]/2
                indexes_stats_time_group_dict[cur_group][cur_time_bin] = intra_country_indexes_stats_time_group_dict[cur_group][cur_time_bin] + inter_country_indexes_stats_time_group_dict[cur_group][cur_time_bin]

                intra_country_indexes_stats_time_group_ref_dict[cur_time_bin] += intra_country_indexes_stats_time_group_dict[cur_group][cur_time_bin]
                inter_country_indexes_stats_time_group_ref_dict[cur_time_bin] += inter_country_indexes_stats_time_group_dict[cur_group][cur_time_bin]
                indexes_stats_time_group_ref_dict[cur_time_bin] += indexes_stats_time_group_dict[cur_group][cur_time_bin]



        # temporal analysis for news article similarity of political groups
        political_group_group_bin_dict_list = defaultdict(lambda: defaultdict(list))
        political_group_group_bin_inter_country_dict_list = defaultdict(lambda: defaultdict(list))

        political_group_group_bin_dict = defaultdict(lambda: defaultdict(float))
        political_group_group_bin_95interval_error = defaultdict(lambda: defaultdict(float))

        political_group_group_sim_bin_for_present = defaultdict(list)
        political_group_group_count_bin_for_present = defaultdict(list)
        political_group_group_bin_95interval_error_for_present = defaultdict(list)
        political_group_group_bin_for_present = defaultdict(list)

        political_group_ref_bin_dict_list = defaultdict(list)
        political_group_ref_sim_bin_for_present = []
        political_group_ref_count_bin_for_present = []
        political_group_ref_sim_bin_95interval_error_for_present = []
        political_group_ref_bin_for_present = []

        political_group_across_group_bin_dict_list = defaultdict(lambda: defaultdict(list))
        political_group_across_group_bin_dict = defaultdict(lambda: defaultdict(float))
        political_group_across_group_bin_95interval_error = defaultdict(lambda: defaultdict(float))


        political_group_across_group_sim_bin_for_present = defaultdict(list)
        political_group_across_group_count_bin_for_present = defaultdict(list)
        political_group_across_group_bin_95interval_error_for_present = defaultdict(list)
        political_group_across_group_bin_for_present = defaultdict(list)

        political_group_across_ref_bin_dict_list = defaultdict(list)
        political_group_across_ref_sim_bin_for_present = []
        political_group_across_ref_count_bin_for_present = []
        political_group_across_ref_bin_for_present = []

        political_group_pair_groups = political_group_pair["political_groups"].to_list()
        political_group_pair_across_groups = political_group_pair["political_across_groups"].to_list()
        political_group_pair_bin = political_group_pair["time_bin"].to_list()
        political_group_pair_sim = political_group_pair["similarity"].to_list()
        political_group_pair_country_full_name1 = political_group_pair["country_full_name1"].to_list()
        political_group_pair_country_full_name2 = political_group_pair["country_full_name2"].to_list()

        # # trying to have similar pair number for each country/country-pair at a certain timestamp
        # political_group_pair_group_bin_country = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

        for i in range(len(political_group_pair_groups)):
            for group in political_group_pair_groups[i]:

                political_group_group_bin_dict_list[group][political_group_pair_bin[i]].append(political_group_pair_sim[i])
                if political_group_pair_country_full_name1[i] != political_group_pair_country_full_name2[i]:
                    political_group_group_bin_inter_country_dict_list[group][political_group_pair_bin[i]].append(political_group_pair_sim[i])
            political_group_ref_bin_dict_list[political_group_pair_bin[i]].append(political_group_pair_sim[i])
        for group in political_group_group_bin_dict_list:
            for bin in political_group_group_bin_dict_list[group]:
                political_group_group_bin_dict[group][bin] = np.mean(political_group_group_bin_dict_list[group][bin])

                temp_low, temp_high = st.t.interval(alpha=0.95,
                                                    df=len(political_group_group_bin_dict_list[group][bin]) - 1,
                                                    loc=np.mean(political_group_group_bin_dict_list[group][bin]),
                                                    scale=st.stats.sem(political_group_group_bin_dict_list[group][bin]))
                political_group_group_bin_95interval_error[group][bin] = political_group_group_bin_dict[group][bin] - temp_low
        for group in political_group_group_bin_dict:
            cur_bins = list(political_group_group_bin_dict_list[group].keys())
            cur_bins.sort()
            for bin in cur_bins:
                political_group_group_sim_bin_for_present[group].append(political_group_group_bin_dict[group][bin])
                political_group_group_count_bin_for_present[group].append(np.log(len(political_group_group_bin_dict_list[group][bin])/indexes_stats_time_group_dict[group][bin]))
                political_group_group_bin_95interval_error_for_present[group].append(political_group_group_bin_95interval_error[group][bin])
            political_group_group_bin_for_present[group] = cur_bins
        ref_bins = list(political_group_ref_bin_dict_list.keys())
        ref_bins.sort()
        for bin in ref_bins:
            political_group_ref_sim_bin_for_present.append(np.mean(political_group_ref_bin_dict_list[bin]))
            political_group_ref_count_bin_for_present.append(np.log(len(political_group_ref_bin_dict_list[bin])/indexes_stats_time_group_ref_dict[bin]))

            temp_low, temp_high = st.t.interval(alpha=0.95,
                                                df=len(political_group_ref_bin_dict_list[bin]) - 1,
                                                loc=np.mean(political_group_ref_bin_dict_list[bin]),
                                                scale=st.stats.sem(political_group_ref_bin_dict_list[bin]))
            political_group_ref_sim_bin_95interval_error_for_present.append(np.mean(political_group_ref_bin_dict_list[bin]) - temp_low)
        political_group_ref_bin_for_present = ref_bins

        for i in range(len(political_group_pair_across_groups)):
            for combo in political_group_pair_across_groups[i]:
                political_group_across_group_bin_dict_list[combo][political_group_pair_bin[i]].append(political_group_pair_sim[i])
            political_group_across_ref_bin_dict_list[political_group_pair_bin[i]].append(political_group_pair_sim[i])
        for combo in political_group_across_group_bin_dict_list:
            for bin in political_group_across_group_bin_dict_list[combo]:
                political_group_across_group_bin_dict[combo][bin] = np.mean(political_group_across_group_bin_dict_list[combo][bin])

                temp_low, temp_high = st.t.interval(alpha=0.95,
                                                    df=len(political_group_across_group_bin_dict_list[combo][bin]) - 1,
                                                    loc=np.mean(political_group_across_group_bin_dict_list[combo][bin]),
                                                    scale=st.stats.sem(political_group_across_group_bin_dict_list[combo][bin]))
                political_group_across_group_bin_95interval_error[combo][bin] = political_group_across_group_bin_dict[combo][bin] - temp_low


        for combo in political_group_across_group_bin_dict:
            cur_bins = list(political_group_across_group_bin_dict_list[combo].keys())
            cur_bins.sort()
            for bin in cur_bins:
                political_group_across_group_sim_bin_for_present[combo].append(political_group_across_group_bin_dict[combo][bin])
                political_group_across_group_count_bin_for_present[combo].append(np.log(len(political_group_across_group_bin_dict_list[combo][bin])))
                political_group_across_group_bin_95interval_error_for_present[combo].append(political_group_across_group_bin_95interval_error[combo][bin])
            political_group_across_group_bin_for_present[combo] = cur_bins
        ref_bins = list(political_group_across_ref_bin_dict_list.keys())
        ref_bins.sort()
        for bin in ref_bins:
            political_group_across_ref_sim_bin_for_present.append(np.mean(political_group_across_ref_bin_dict_list[bin]))
            political_group_across_ref_count_bin_for_present.append(np.log(len(political_group_across_ref_bin_dict_list[bin])))
        political_group_across_ref_bin_for_present = ref_bins

        # compute correlation for each political group and across political group
        political_group_group_bin_coef_dict_list = defaultdict(list)
        political_group_group_sim_coef_dict_list = defaultdict(list)
        political_group_across_group_bin_coef_dict_list = defaultdict(list)
        political_group_across_group_sim_coef_dict_list = defaultdict(list)
        for group in political_group_group_bin_dict_list:
            for bin in political_group_group_bin_dict_list[group]:
                for pair_sim in political_group_group_bin_dict_list[group][bin]:
                    political_group_group_bin_coef_dict_list[group].append(bin)
                    political_group_group_sim_coef_dict_list[group].append(pair_sim)
        for combo in political_group_across_group_bin_dict_list:
            for bin in political_group_across_group_bin_dict_list[combo]:
                for pair_sim in political_group_across_group_bin_dict_list[combo][bin]:
                    political_group_across_group_bin_coef_dict_list[combo].append(bin)
                    political_group_across_group_sim_coef_dict_list[combo].append(pair_sim)

        for group in political_group_group_sim_coef_dict_list:
            print(f"temporal analysis: within {group} political group pairs pearson correlation and p-value are:", st.pearsonr(political_group_group_bin_coef_dict_list[group], political_group_group_sim_coef_dict_list[group]))
            print(f"temporal analysis: within {group} political group pairs spearman correlation and p-value are:", st.spearmanr(political_group_group_bin_coef_dict_list[group], political_group_group_sim_coef_dict_list[group]))
        for combo in political_group_across_group_sim_coef_dict_list:
            print(f"temporal analysis: within {combo} across political group pairs pearson correlation and p-value are:", st.pearsonr(political_group_across_group_bin_coef_dict_list[combo], political_group_across_group_sim_coef_dict_list[combo]))
            print(f"temporal analysis: within {combo} across political group pairs spearman correlation and p-value are:", st.spearmanr(political_group_across_group_bin_coef_dict_list[combo], political_group_across_group_sim_coef_dict_list[combo]))

        political_group_color = {"nato":"blue","eunion":"orange","brics":"green"}
        fig = plt.figure(figsize=(10, 8))
        for group in political_group_group_sim_bin_for_present:
            # plt.plot(political_group_group_bin_for_present[group], political_group_group_sim_bin_for_present[group], marker="o")
            plt.plot(political_group_group_bin_for_present[group][:12], political_group_group_sim_bin_for_present[group][:12], color=political_group_color[group], marker="o")
            plt.errorbar(political_group_group_bin_for_present[group][:12], political_group_group_sim_bin_for_present[group][:12],
                         color=political_group_color[group],
                         yerr=[political_group_group_bin_95interval_error_for_present[group][:12],
                               political_group_group_bin_95interval_error_for_present[group][:12]], fmt='o',
                         markersize=8, capsize=20)

        plt.plot(political_group_ref_bin_for_present[:12], political_group_ref_sim_bin_for_present[:12], marker="o", color="black")
        plt.errorbar(political_group_ref_bin_for_present[:12], political_group_ref_sim_bin_for_present[:12], color="black", yerr=[political_group_ref_sim_bin_95interval_error_for_present[:12], political_group_ref_sim_bin_95interval_error_for_present[:12]], fmt='o', markersize=8, capsize=20)


        # plt.xticks(ticks=range(len(political_group_group_bin_for_present[group])), labels=month_label, fontsize=20)
        plt.xticks(ticks=range(12), labels=['', 'Jan', '', 'Feb', '', 'Mar', '', 'Apr', '', 'May', '', 'Jun'], fontsize=20)
        plt.yticks(size=20)
        plt.xlabel('Time', fontsize=32)
        plt.ylabel('Similarity', fontsize=32)
        cur_legend = list(political_group_group_sim_bin_for_present.keys())
        cur_legend.append("world")
        plt.legend(cur_legend, fontsize=20)

        # plt.show()
        plt.savefig(f"{output_dir}/time_sim_political_groups.png")


        fig = plt.figure(figsize=(10, 8))
        for group in political_group_group_count_bin_for_present:
            # plt.plot(political_group_group_bin_for_present[group], political_group_group_sim_bin_for_present[group], marker="o")
            plt.plot(political_group_group_bin_for_present[group][:12], political_group_group_count_bin_for_present[group][:12], marker="o")
        plt.plot(political_group_ref_bin_for_present[:12], political_group_ref_count_bin_for_present[:12], marker="o")

        # plt.xticks(ticks=range(len(political_group_group_bin_for_present[group])), labels=month_label, fontsize=20)
        plt.xticks(ticks=range(12), labels=['', 'Jan', '', 'Feb', '', 'Mar', '', 'Apr', '', 'May', '', 'Jun'],
                   fontsize=20)
        plt.yticks(size=20)
        plt.xlabel('Time', fontsize=32)
        plt.ylabel('Fraction', fontsize=32)
        cur_legend = list(political_group_group_count_bin_for_present.keys())
        cur_legend.append("world")
        plt.legend(cur_legend, fontsize=20)

        # plt.show()
        plt.savefig(f"{output_dir}/time_frac_political_groups.png")


        across_political_group_color = {"nato-eunion":"blue", "eunion-brics":"orange", "nato-brics":"green"}
        # plot pairs across political group
        fig = plt.figure(figsize=(10, 8))
        for combo in political_group_across_group_sim_bin_for_present:
            # plt.plot(political_group_across_group_bin_for_present[combo], political_group_across_group_sim_bin_for_present[combo], marker="o")
            plt.plot(political_group_across_group_bin_for_present[combo][:12], political_group_across_group_sim_bin_for_present[combo][:12], marker="o",color=across_political_group_color[combo])
            plt.errorbar(political_group_across_group_bin_for_present[combo][:12], political_group_across_group_sim_bin_for_present[combo][:12], color=across_political_group_color[combo], yerr=[political_group_across_group_bin_95interval_error_for_present[combo][:12], political_group_across_group_bin_95interval_error_for_present[combo][:12]], fmt='o', markersize=8, capsize=20)
        # plt.plot(political_group_across_ref_bin_for_present[:12], political_group_across_ref_sim_bin_for_present[:12], marker="o")


        plt.xticks(ticks=range(12), labels=['', 'Jan', '', 'Feb', '', 'Mar', '', 'Apr', '', 'May', '', 'Jun'], fontsize=20)
        plt.yticks(size=20)
        plt.xlabel('Time', fontsize=32)
        plt.ylabel('Similarity', fontsize=32)
        plt.legend(list(political_group_across_group_sim_bin_for_present.keys()), fontsize=20)

        # plt.show()
        plt.savefig(f"{output_dir}/time_sim_across_political_groups.png")

        fig = plt.figure(figsize=(10, 8))
        for combo in political_group_across_group_count_bin_for_present:
            # plt.plot(political_group_across_group_bin_for_present[combo], political_group_across_group_sim_bin_for_present[combo], marker="o")
            plt.plot(political_group_across_group_bin_for_present[combo][:12], political_group_across_group_count_bin_for_present[combo][:12], marker="o")
        plt.plot(political_group_across_ref_bin_for_present[:12], political_group_across_ref_count_bin_for_present[:12], marker="o")

        plt.xticks(ticks=range(12), labels=['', 'Jan', '', 'Feb', '', 'Mar', '', 'Apr', '', 'May', '', 'Jun'],
                   fontsize=20)
        plt.yticks(size=20)
        plt.xlabel('Time', fontsize=32)
        plt.ylabel('Pair Count', fontsize=32)
        cur_legend = list(political_group_across_group_count_bin_for_present.keys())
        cur_legend.append("world")
        plt.legend(cur_legend, fontsize=20)

        # plt.show()
        plt.savefig(f"{output_dir}/time_count_across_political_groups.png")


        # plot the likelyhood across political group pairs are larger than within political group pairs
        # the rough idea is: taking three articles in month m, two from political block X (x1 and x2) and one from Y (y),
        # compute the likelihood of sim(x1, y) > sim(x1, x2). if likelihood decreases over time, we can interpret this as X and Y coming apart
        # need to fix, we shouldn't compute the intra-country pairs

        print("before computing likelyhood: ", datetime.now())

        across_group_positive_num = defaultdict(lambda: defaultdict(int))
        across_group_diff = defaultdict(lambda: defaultdict(list))
        across_group_total_num = defaultdict(lambda: defaultdict(int))
        across_group_diff_mean = defaultdict(lambda: defaultdict(float))
        across_group_likelyhood = defaultdict(lambda: defaultdict(float))
        across_group_diff_mean_for_present = defaultdict(list)
        across_group_likelyhood_for_present = defaultdict(list)
        across_group_bin_for_present = defaultdict(list)
        for combo in political_group_across_group_bin_dict_list:
            cur_group1, cur_group2 = combo.split("-")
            for bin in political_group_across_group_bin_dict_list[combo]:
                for across_group_sim in political_group_across_group_bin_dict_list[combo][bin]:
                    for within_group_sim in political_group_group_bin_inter_country_dict_list[cur_group1][bin]:
                        if across_group_sim > within_group_sim:
                            across_group_positive_num[combo][bin] += 1
                        across_group_diff[combo][bin].append(across_group_sim-within_group_sim)
                        across_group_total_num[combo][bin] += 1

                    for within_group_sim in political_group_group_bin_dict_list[cur_group2][bin]:
                        if across_group_sim > within_group_sim:
                            across_group_positive_num[combo][bin] += 1
                        across_group_diff[combo][bin].append(across_group_sim-within_group_sim)
                        across_group_total_num[combo][bin] += 1

                    if across_group_total_num[combo][bin] > 500000:
                        break

                across_group_diff_mean[combo][bin] = np.mean(across_group_diff[combo][bin])/across_group_total_num[combo][bin]
                across_group_likelyhood[combo][bin] = across_group_positive_num[combo][bin]/across_group_total_num[combo][bin]


        for combo in across_group_likelyhood:
            cur_bins = list(across_group_likelyhood[combo].keys())
            cur_bins.sort()
            for bin in across_group_likelyhood[combo]:
                across_group_diff_mean_for_present[combo].append(across_group_diff_mean[combo][bin])
                across_group_likelyhood_for_present[combo].append(across_group_likelyhood[combo][bin])
                across_group_bin_for_present[combo].append(bin)

        print("after computing likelyhood: ", datetime.now())

        fig = plt.figure(figsize=(10, 8))
        for combo in across_group_diff_mean_for_present:
            # plt.plot(intra_country_political_group_group_bin_for_present[group], intra_country_political_group_group_sim_bin_for_present[group], marker="o")
            plt.plot(across_group_bin_for_present[combo][:12], across_group_diff_mean_for_present[combo][:12],
                     marker="o")

        # plt.xticks(ticks=range(len(intra_country_political_group_group_bin_for_present[group])), labels=month_label, fontsize=20)
        plt.xticks(ticks=range(12), labels=['', 'Jan', '', 'Feb', '', 'Mar', '', 'Apr', '', 'May', '', 'Jun'],
                   fontsize=20)
        plt.yticks(size=20)
        plt.xlabel('Time', fontsize=32)
        plt.ylabel('Similarity Difference', fontsize=32)
        cur_legend = list(across_group_likelyhood_for_present.keys())
        plt.legend(cur_legend, fontsize=20)

        # plt.show()
        plt.savefig(f"{output_dir}/time_sim_diff_across_political_groups.png")

        fig = plt.figure(figsize=(10, 8))
        for combo in across_group_likelyhood_for_present:
            # plt.plot(intra_country_political_group_group_bin_for_present[group], intra_country_political_group_group_sim_bin_for_present[group], marker="o")
            plt.plot(across_group_bin_for_present[combo][:12], across_group_likelyhood_for_present[combo][:12], marker="o")

        # plt.xticks(ticks=range(len(intra_country_political_group_group_bin_for_present[group])), labels=month_label, fontsize=20)
        plt.xticks(ticks=range(12), labels=['', 'Jan', '', 'Feb', '', 'Mar', '', 'Apr', '', 'May', '', 'Jun'],
                   fontsize=20)
        plt.yticks(size=20)
        plt.xlabel('Time', fontsize=32)
        plt.ylabel('Likelihood', fontsize=32)
        cur_legend = list(across_group_likelyhood_for_present.keys())
        plt.legend(cur_legend, fontsize=20)

        # plt.show()
        plt.savefig(f"{output_dir}/time_likelyhood_across_political_groups.png")





        # focse on intra-country pairs: temporal analysis for news article similarity of political groups
        intra_country_political_group_pair = political_group_pair.loc[(political_group_pair["main_country1"] == political_group_pair["main_country2"])]

        intra_country_political_group_group_bin_dict_list = defaultdict(lambda: defaultdict(list))
        intra_country_political_group_group_bin_dict = defaultdict(lambda: defaultdict(float))
        intra_country_political_group_group_sim_bin_for_present = defaultdict(list)
        intra_country_political_group_group_count_bin_for_present = defaultdict(list)
        intra_country_political_group_group_bin_for_present = defaultdict(list)

        intra_country_political_group_ref_bin_dict_list = defaultdict(list)
        intra_country_political_group_ref_sim_bin_for_present = []
        intra_country_political_group_ref_count_bin_for_present = []
        intra_country_political_group_ref_bin_for_present = []


        intra_country_political_group_pair_groups = intra_country_political_group_pair["political_groups"].to_list()
        intra_country_political_group_pair_bin = intra_country_political_group_pair["time_bin"].to_list()
        intra_country_political_group_pair_sim = intra_country_political_group_pair["similarity"].to_list()
        for i in range(len(intra_country_political_group_pair_groups)):
            for group in intra_country_political_group_pair_groups[i]:
                intra_country_political_group_group_bin_dict_list[group][intra_country_political_group_pair_bin[i]].append(intra_country_political_group_pair_sim[i])
            intra_country_political_group_ref_bin_dict_list[intra_country_political_group_pair_bin[i]].append(intra_country_political_group_pair_sim[i])
        for group in intra_country_political_group_group_bin_dict_list:
            for bin in intra_country_political_group_group_bin_dict_list[group]:
                intra_country_political_group_group_bin_dict[group][bin] = np.mean(intra_country_political_group_group_bin_dict_list[group][bin])
        for group in intra_country_political_group_group_bin_dict:
            cur_bins = list(intra_country_political_group_group_bin_dict_list[group].keys())
            cur_bins.sort()
            for bin in cur_bins:
                intra_country_political_group_group_sim_bin_for_present[group].append(intra_country_political_group_group_bin_dict[group][bin])
                intra_country_political_group_group_count_bin_for_present[group].append(np.log(len(intra_country_political_group_group_bin_dict_list[group][bin])/intra_country_indexes_stats_time_group_dict[group][bin]))
            intra_country_political_group_group_bin_for_present[group] = cur_bins
        ref_bins = list(intra_country_political_group_ref_bin_dict_list.keys())
        ref_bins.sort()
        for bin in ref_bins:
            intra_country_political_group_ref_sim_bin_for_present.append(np.mean(intra_country_political_group_ref_bin_dict_list[bin]))
            intra_country_political_group_ref_count_bin_for_present.append(np.log(len(intra_country_political_group_ref_bin_dict_list[bin])/intra_country_indexes_stats_time_group_ref_dict[bin]))
        intra_country_political_group_ref_bin_for_present = ref_bins

        fig = plt.figure(figsize=(10, 8))
        for group in intra_country_political_group_group_sim_bin_for_present:
            # plt.plot(intra_country_political_group_group_bin_for_present[group], intra_country_political_group_group_sim_bin_for_present[group], marker="o")
            plt.plot(intra_country_political_group_group_bin_for_present[group][:12], intra_country_political_group_group_sim_bin_for_present[group][:12], marker="o")
        plt.plot(intra_country_political_group_ref_bin_for_present[:12], intra_country_political_group_ref_sim_bin_for_present[:12], marker="o")

        # plt.xticks(ticks=range(len(intra_country_political_group_group_bin_for_present[group])), labels=month_label, fontsize=20)
        plt.xticks(ticks=range(12), labels=['', 'Jan', '', 'Feb', '', 'Mar', '', 'Apr', '', 'May', '', 'Jun'],
                   fontsize=20)
        plt.yticks(size=20)
        plt.xlabel('Time', fontsize=32)
        plt.ylabel('Similarity', fontsize=32)
        cur_legend = list(intra_country_political_group_group_sim_bin_for_present.keys())
        cur_legend.append("world")
        plt.legend(cur_legend, fontsize=20)

        # plt.show()
        plt.savefig(f"{output_dir}/time_sim_political_groups_intra_country.png")




        fig = plt.figure(figsize=(10, 8))
        for group in intra_country_political_group_group_count_bin_for_present:
            # plt.plot(intra_country_political_group_group_bin_for_present[group], intra_country_political_group_group_sim_bin_for_present[group], marker="o")
            plt.plot(intra_country_political_group_group_bin_for_present[group][:12], intra_country_political_group_group_count_bin_for_present[group][:12], marker="o")
        plt.plot(intra_country_political_group_ref_bin_for_present[:12], intra_country_political_group_ref_count_bin_for_present[:12], marker="o")

        # plt.xticks(ticks=range(len(intra_country_political_group_group_bin_for_present[group])), labels=month_label, fontsize=20)
        plt.xticks(ticks=range(12), labels=['', 'Jan', '', 'Feb', '', 'Mar', '', 'Apr', '', 'May', '', 'Jun'],
                   fontsize=20)
        plt.yticks(size=20)
        plt.xlabel('Time', fontsize=32)
        plt.ylabel('Fraction', fontsize=32)
        cur_legend = list(intra_country_political_group_group_count_bin_for_present.keys())
        cur_legend.append("world")
        plt.legend(cur_legend, fontsize=20)

        # plt.show()
        plt.savefig(f"{output_dir}/time_frac_political_groups_intra_country.png")

        # focse on inter-country pairs: temporal analysis for news article similarity of political groups
        inter_country_political_group_pair = political_group_pair.loc[
            political_group_pair["main_country1"] != political_group_pair["main_country2"]]

        inter_country_political_group_group_bin_dict_list = defaultdict(lambda: defaultdict(list))
        inter_country_political_group_group_bin_dict = defaultdict(lambda: defaultdict(float))
        inter_country_political_group_group_sim_bin_for_present = defaultdict(list)
        inter_country_political_group_group_count_bin_for_present = defaultdict(list)
        inter_country_political_group_group_bin_for_present = defaultdict(list)

        inter_country_political_group_95interval_error_bin_for_present = defaultdict(list)

        inter_country_political_group_ref_bin_dict_list = defaultdict(list)
        inter_country_political_group_ref_sim_bin_for_present = []
        inter_country_political_group_ref_count_bin_for_present = []
        inter_country_political_group_ref_bin_for_present = []

        inter_country_political_group_ref_95interval_error_bin_for_present = []

        inter_country_political_group_pair_groups = inter_country_political_group_pair["political_groups"].to_list()
        inter_country_political_group_pair_bin = inter_country_political_group_pair["time_bin"].to_list()
        inter_country_political_group_pair_sim = inter_country_political_group_pair["similarity"].to_list()
        for i in range(len(inter_country_political_group_pair_groups)):
            for group in inter_country_political_group_pair_groups[i]:
                inter_country_political_group_group_bin_dict_list[group][inter_country_political_group_pair_bin[i]].append(inter_country_political_group_pair_sim[i])
            inter_country_political_group_ref_bin_dict_list[inter_country_political_group_pair_bin[i]].append(inter_country_political_group_pair_sim[i])
        for group in inter_country_political_group_group_bin_dict_list:
            for bin in inter_country_political_group_group_bin_dict_list[group]:
                inter_country_political_group_group_bin_dict[group][bin] = np.mean(inter_country_political_group_group_bin_dict_list[group][bin])
        for group in inter_country_political_group_group_bin_dict:
            cur_bins = list(inter_country_political_group_group_bin_dict_list[group].keys())
            cur_bins.sort()
            for bin in cur_bins:
                inter_country_political_group_group_sim_bin_for_present[group].append(inter_country_political_group_group_bin_dict[group][bin])
                inter_country_political_group_group_count_bin_for_present[group].append(np.log10(len(inter_country_political_group_group_bin_dict_list[group][bin])/inter_country_indexes_stats_time_group_dict[group][bin]))

                temp_low, temp_high = st.t.interval(alpha=0.95,
                                                    df=len(inter_country_political_group_group_bin_dict_list[group][bin]) - 1,
                                                    loc=np.mean(inter_country_political_group_group_bin_dict_list[group][bin]),
                                                    scale=st.stats.sem(inter_country_political_group_group_bin_dict_list[group][bin]))
                inter_country_political_group_95interval_error_bin_for_present[group].append(np.mean(inter_country_political_group_group_bin_dict_list[group][bin]) - temp_low)
            inter_country_political_group_group_bin_for_present[group] = cur_bins
        ref_bins = list(inter_country_political_group_ref_bin_dict_list.keys())
        ref_bins.sort()
        for bin in ref_bins:
            inter_country_political_group_ref_sim_bin_for_present.append(np.mean(inter_country_political_group_ref_bin_dict_list[bin]))
            inter_country_political_group_ref_count_bin_for_present.append(np.log10(len(inter_country_political_group_ref_bin_dict_list[bin])/inter_country_indexes_stats_time_group_ref_dict[bin]))

            temp_low, temp_high = st.t.interval(alpha=0.95, df=len(inter_country_political_group_ref_bin_dict_list[bin]) - 1,
                                                loc=np.mean(inter_country_political_group_ref_bin_dict_list[bin]),
                                                scale=st.stats.sem(inter_country_political_group_ref_bin_dict_list[bin]))
            inter_country_political_group_ref_95interval_error_bin_for_present.append(np.mean(inter_country_political_group_ref_bin_dict_list[bin]) - temp_low)
        inter_country_political_group_ref_bin_for_present = ref_bins

        # compute correlation
        print("temporal analysis: inter-country pairs pearson correlation and p-value are:", st.pearsonr(inter_country_political_group_pair_bin, inter_country_political_group_pair_sim))
        print("temporal analysis: inter-country pairs spearman correlation and p-value are:", st.spearmanr(inter_country_political_group_pair_bin, inter_country_political_group_pair_sim))

        fig = plt.figure(figsize=(10, 8))
        for group in inter_country_political_group_group_sim_bin_for_present:
            # plt.plot(inter_country_political_group_group_bin_for_present[group], inter_country_political_group_group_sim_bin_for_present[group], marker="o")
            plt.plot(inter_country_political_group_group_bin_for_present[group][:12], inter_country_political_group_group_sim_bin_for_present[group][:12], color=political_group_color[group], marker="o")
            plt.errorbar(inter_country_political_group_group_bin_for_present[group][:12], inter_country_political_group_group_sim_bin_for_present[group][:12], yerr=[inter_country_political_group_95interval_error_bin_for_present[group][:12], inter_country_political_group_95interval_error_bin_for_present[group][:12]], color=political_group_color[group], fmt='o', markersize=8, capsize=20)

        # plt.plot(inter_country_political_group_ref_bin_for_present[:12], inter_country_political_group_ref_sim_bin_for_present[:12], color="purple", marker="o")
        # plt.errorbar(inter_country_political_group_ref_bin_for_present[:12], inter_country_political_group_ref_sim_bin_for_present[:12], color="purple", yerr=[inter_country_political_group_ref_95interval_error_bin_for_present[:12], inter_country_political_group_ref_95interval_error_bin_for_present[:12]], fmt='o', markersize=8, capsize=20)

        # plt.xticks(ticks=range(len(inter_country_political_group_group_bin_for_present[group])), labels=month_label, fontsize=20)
        plt.xticks(ticks=range(12), labels=['', 'Jan', '', 'Feb', '', 'Mar', '', 'Apr', '', 'May', '', 'Jun'],
                   fontsize=20)
        plt.yticks(size=20)
        plt.xlabel('Time', fontsize=32)
        plt.ylabel('Similarity', fontsize=32)
        cur_legend = list(inter_country_political_group_group_sim_bin_for_present.keys())
        # cur_legend.append("world")
        plt.legend(cur_legend, fontsize=20)

        # plt.show()
        plt.savefig(f"{output_dir}/time_sim_political_groups_inter_country.png")

        fig = plt.figure(figsize=(10, 8))
        for group in inter_country_political_group_group_sim_bin_for_present:
            # plt.plot(inter_country_political_group_group_bin_for_present[group], inter_country_political_group_group_sim_bin_for_present[group], marker="o")
            plt.plot(inter_country_political_group_group_bin_for_present[group][:12], inter_country_political_group_group_count_bin_for_present[group][:12], marker="o")
        plt.plot(inter_country_political_group_ref_bin_for_present[:12], inter_country_political_group_ref_count_bin_for_present[:12], marker="o")

        # plt.xticks(ticks=range(len(inter_country_political_group_group_bin_for_present[group])), labels=month_label, fontsize=20)
        plt.xticks(ticks=range(12), labels=['', 'Jan', '', 'Feb', '', 'Mar', '', 'Apr', '', 'May', '', 'Jun'],
                   fontsize=20)
        plt.yticks(size=20)
        plt.xlabel('Time', fontsize=32)
        plt.ylabel('Fraction', fontsize=32)
        cur_legend = list(inter_country_political_group_group_count_bin_for_present.keys())
        cur_legend.append("world")
        plt.legend(cur_legend, fontsize=20)

        # plt.show()
        plt.savefig(f"{output_dir}/time_frac_political_groups_inter_country.png")


        # plotting similairty as per countries within each political gorup (no fraction so far or needed)
        country_color = ["blue",'green','red','orange','yellow','grey','purple', 'silver','pink', 'wheat','coral','peru','lightgreen','skyblue','navy']

        uni_country_within_political_group_pair = political_group_pair.loc[political_group_pair["main_country1"] == political_group_pair["main_country2"]]
        uni_country_within_political_group_pair = uni_country_within_political_group_pair.loc[political_group_pair["political_groups_num"] == 1]
        uni_country_within_political_group_pair["uni_political_groups"] = uni_country_within_political_group_pair.apply(lambda x: x["political_groups"][0], axis=1)
        uni_country_within_each_political_group_dict = {}
        for group in political_group_list:
            uni_country_within_each_political_group_dict[group] = uni_country_within_political_group_pair.loc[uni_country_within_political_group_pair["uni_political_groups"] == group]
        # uni_country_within_nato_pair = uni_country_within_political_group_pair.loc[uni_country_within_political_group_pair["uni_political_groups"] == 'nato']
        # uni_country_within_eunion_pair = uni_country_within_political_group_pair.loc[uni_country_within_political_group_pair["uni_political_groups"] == 'eunion']
        # uni_country_within_brics_pair = uni_country_within_political_group_pair.loc[uni_country_within_political_group_pair["uni_political_groups"] == 'brics']

        for group in political_group_list:
            uni_country_within_each_political_group_group_bin_dict_list = defaultdict(lambda: defaultdict(list))
            uni_country_within_each_political_group_group_bin_dict = defaultdict(lambda: defaultdict(float))
            uni_country_within_each_political_group_group_sim_bin_for_present = defaultdict(list)
            uni_country_within_each_political_group_group_count_bin_for_present = defaultdict(list)
            uni_country_within_each_political_group_group_bin_for_present = defaultdict(list)
            uni_country_within_each_political_group_95interval_error_bin_for_present = defaultdict(list)

            uni_country_within_each_political_group_ref_bin_dict_list = defaultdict(list)
            uni_country_within_each_political_group_ref_sim_bin_for_present = []
            uni_country_within_each_political_group_ref_count_bin_for_present = []
            uni_country_within_each_political_group_ref_bin_for_present = []
            uni_country_within_each_political_group_ref_95interval_error_bin_for_present = []

            remove_uni_country_within_each_political_group_group_bin_dict_list = defaultdict(lambda: defaultdict(list))
            remove_uni_country_within_each_political_group_group_sim_bin_for_present = defaultdict(list)
            remove_uni_country_within_each_political_group_95interval_error_bin_for_present = defaultdict(list)

            uni_country_within_each_political_group_pair_groups = uni_country_within_each_political_group_dict[group]["main_country1"].to_list()
            uni_country_within_each_political_group_pair_bin = uni_country_within_each_political_group_dict[group]["time_bin"].to_list()
            uni_country_within_each_political_group_pair_sim = uni_country_within_each_political_group_dict[group]["similarity"].to_list()
            for i in range(len(uni_country_within_each_political_group_pair_groups)):
                country = uni_country_within_each_political_group_pair_groups[i]
                uni_country_within_each_political_group_group_bin_dict_list[country][uni_country_within_each_political_group_pair_bin[i]].append(uni_country_within_each_political_group_pair_sim[i])
                uni_country_within_each_political_group_ref_bin_dict_list[uni_country_within_each_political_group_pair_bin[i]].append(uni_country_within_each_political_group_pair_sim[i])
            for country in uni_country_within_each_political_group_group_bin_dict_list:
                for bin in uni_country_within_each_political_group_group_bin_dict_list[country]:
                    uni_country_within_each_political_group_group_bin_dict[country][bin] = np.mean(uni_country_within_each_political_group_group_bin_dict_list[country][bin])

                    temp_low, temp_high = st.t.interval(alpha=0.95,df=len(uni_country_within_each_political_group_group_bin_dict_list[country][bin]) - 1, loc=np.mean(uni_country_within_each_political_group_group_bin_dict_list[country][bin]), scale=st.stats.sem(uni_country_within_each_political_group_group_bin_dict_list[country][bin]))
                    uni_country_within_each_political_group_95interval_error_bin_for_present[country].append(np.mean(uni_country_within_each_political_group_group_bin_dict_list[country][bin]) - temp_low)

                    print(f"{country} at time bin {bin} has ", len(uni_country_within_each_political_group_group_bin_dict_list[country][bin]), " pairs...")

                    for other_country in uni_country_within_each_political_group_group_bin_dict_list:
                        if (other_country != country) and (bin in uni_country_within_each_political_group_group_bin_dict_list[other_country]):
                            remove_uni_country_within_each_political_group_group_bin_dict_list[country][bin] += uni_country_within_each_political_group_group_bin_dict_list[other_country][bin]



            for country in uni_country_within_each_political_group_group_bin_dict:
                cur_bins = list(uni_country_within_each_political_group_group_bin_dict_list[country].keys())
                cur_bins.sort()
                for bin in cur_bins:
                    uni_country_within_each_political_group_group_sim_bin_for_present[country].append(uni_country_within_each_political_group_group_bin_dict[country][bin])
                    uni_country_within_each_political_group_group_count_bin_for_present[country].append(np.log10(len(uni_country_within_each_political_group_group_bin_dict_list[country][bin])/indexes_stats_time_country_dict[country][bin]))
                uni_country_within_each_political_group_group_bin_for_present[country] = cur_bins
            ref_bins = list(uni_country_within_each_political_group_ref_bin_dict_list.keys())
            ref_bins.sort()
            for bin in ref_bins:
                uni_country_within_each_political_group_ref_sim_bin_for_present.append(np.mean(uni_country_within_each_political_group_ref_bin_dict_list[bin]))
                uni_country_within_each_political_group_ref_count_bin_for_present.append(np.log10(len(uni_country_within_each_political_group_ref_bin_dict_list[bin])/indexes_stats_time_country_ref_dict[group][bin]))

                temp_low, temp_high = st.t.interval(alpha=0.95, df=len(
                    uni_country_within_each_political_group_ref_bin_dict_list[bin]) - 1, loc=np.mean(
                    uni_country_within_each_political_group_ref_bin_dict_list[bin]), scale=st.stats.sem(
                    uni_country_within_each_political_group_ref_bin_dict_list[bin]))
                uni_country_within_each_political_group_ref_95interval_error_bin_for_present.append(np.mean(uni_country_within_each_political_group_ref_bin_dict_list[bin]) - temp_low)

            uni_country_within_each_political_group_ref_bin_for_present = ref_bins

            for country in remove_uni_country_within_each_political_group_group_bin_dict_list:
                cur_bins = list(remove_uni_country_within_each_political_group_group_bin_dict_list[country].keys())
                cur_bins.sort()
                for bin in cur_bins:
                    remove_uni_country_within_each_political_group_group_sim_bin_for_present[country].append(np.mean(remove_uni_country_within_each_political_group_group_bin_dict_list[country][bin]))

                    temp_low, temp_high = st.t.interval(alpha=0.95, df=len(
                        remove_uni_country_within_each_political_group_group_bin_dict_list[country][bin]) - 1, loc=np.mean(
                        remove_uni_country_within_each_political_group_group_bin_dict_list[country][bin]), scale=st.stats.sem(
                        remove_uni_country_within_each_political_group_group_bin_dict_list[country][bin]))
                    remove_uni_country_within_each_political_group_95interval_error_bin_for_present[country].append(np.mean(remove_uni_country_within_each_political_group_group_bin_dict_list[country][bin]) - temp_low)




            # remove taiwan from country list in the figure since it's controversial, Brazil's official language is not covered, south africa don't have enough data
            uni_country_within_each_political_group_group_sim_bin_for_present.pop('TWN', None)
            uni_country_within_each_political_group_group_sim_bin_for_present.pop('BRA', None)
            uni_country_within_each_political_group_group_sim_bin_for_present.pop('ZAF', None)
            remove_uni_country_within_each_political_group_group_sim_bin_for_present.pop('TWN', None)
            remove_uni_country_within_each_political_group_group_sim_bin_for_present.pop('BRA', None)
            remove_uni_country_within_each_political_group_group_sim_bin_for_present.pop('ZAF', None)
            fig = plt.figure(figsize=(10, 8))

            cur_country_idx = 0
            for country in uni_country_within_each_political_group_group_sim_bin_for_present:
                # plt.plot(uni_country_within_each_political_group_group_bin_for_present[group], uni_country_within_each_political_group_group_sim_bin_for_present[group], marker="o")
                plt.plot(uni_country_within_each_political_group_group_bin_for_present[country][:12], uni_country_within_each_political_group_group_sim_bin_for_present[country][:12],color=country_color[cur_country_idx], marker="o")
                plt.errorbar(uni_country_within_each_political_group_group_bin_for_present[country][:12],uni_country_within_each_political_group_group_sim_bin_for_present[country][:12],color=country_color[cur_country_idx],
                             yerr=[uni_country_within_each_political_group_95interval_error_bin_for_present[country][:12],
                                   uni_country_within_each_political_group_95interval_error_bin_for_present[country][:12]], fmt='o', markersize=8, capsize=20)

                cur_country_idx += 1
            plt.plot(uni_country_within_each_political_group_ref_bin_for_present[:12], uni_country_within_each_political_group_ref_sim_bin_for_present[:12], color='black', marker="o")
            plt.errorbar(uni_country_within_each_political_group_ref_bin_for_present[:12],
                         uni_country_within_each_political_group_ref_sim_bin_for_present[:12],
                         yerr=[uni_country_within_each_political_group_ref_95interval_error_bin_for_present[:12],
                               uni_country_within_each_political_group_ref_95interval_error_bin_for_present[:12]], color='black', fmt='o', markersize=8, capsize=20)

            # plt.xticks(ticks=range(len(uni_country_within_each_political_group_group_bin_for_present[group])), labels=month_label, fontsize=20)
            plt.xticks(ticks=range(12), labels=['', 'Jan', '', 'Feb', '', 'Mar', '', 'Apr', '', 'May', '', 'Jun'],
                       fontsize=20)
            plt.yticks(size=20)
            plt.xlabel('Time', fontsize=32)
            plt.ylabel('Similarity', fontsize=32)
            cur_legend =[]
            for country_alpha3 in list(uni_country_within_each_political_group_group_sim_bin_for_present.keys()):
                cur_legend.append(country_alpha3_geography[country_alpha3]['country_full_name'])
            cur_legend.append(group.upper())
            plt.legend(cur_legend, fontsize=16)

            # plt.show()
            plt.savefig(f"{output_dir}/time_sim_{group}_uni_country_within_each.png")

            # remove taiwan from country list in the figure since it's controversial
            uni_country_within_each_political_group_group_sim_bin_for_present.pop('TWN', None)
            uni_country_within_each_political_group_group_sim_bin_for_present.pop('BRA', None)
            uni_country_within_each_political_group_group_sim_bin_for_present.pop('ZAF', None)
            remove_uni_country_within_each_political_group_group_sim_bin_for_present.pop('TWN', None)
            remove_uni_country_within_each_political_group_group_sim_bin_for_present.pop('BRA', None)
            remove_uni_country_within_each_political_group_group_sim_bin_for_present.pop('ZAF', None)
            fig = plt.figure(figsize=(10, 8))
            for country in uni_country_within_each_political_group_group_count_bin_for_present:
                # plt.plot(uni_country_within_each_political_group_group_bin_for_present[group], uni_country_within_each_political_group_group_sim_bin_for_present[group], marker="o")
                plt.plot(uni_country_within_each_political_group_group_bin_for_present[country][:12],
                         uni_country_within_each_political_group_group_count_bin_for_present[country][:12], marker="o")
            plt.plot(uni_country_within_each_political_group_ref_bin_for_present[:12],
                     uni_country_within_each_political_group_ref_count_bin_for_present[:12], marker="o")

            # plt.xticks(ticks=range(len(uni_country_within_each_political_group_group_bin_for_present[group])), labels=month_label, fontsize=20)
            plt.xticks(ticks=range(12), labels=['', 'Jan', '', 'Feb', '', 'Mar', '', 'Apr', '', 'May', '', 'Jun'],
                       fontsize=20)
            plt.yticks(size=20)
            plt.xlabel('Time', fontsize=32)
            plt.ylabel('Fraction', fontsize=32)
            cur_legend =[]
            for country_alpha3 in list(uni_country_within_each_political_group_group_count_bin_for_present.keys()):
                cur_legend.append(country_alpha3_geography[country_alpha3]['country_full_name'])
            cur_legend.append(group.upper())
            plt.legend(cur_legend, fontsize=16)

            # plt.show()
            plt.savefig(f"{output_dir}/time_frac_{group}_uni_country_within_each.png")

            # temporal anlaysis of each political group without any 1 of the in-group countries
            uni_country_within_each_political_group_group_sim_bin_for_present.pop('TWN', None)
            uni_country_within_each_political_group_group_sim_bin_for_present.pop('BRA', None)
            uni_country_within_each_political_group_group_sim_bin_for_present.pop('ZAF', None)
            remove_uni_country_within_each_political_group_group_sim_bin_for_present.pop('TWN', None)
            remove_uni_country_within_each_political_group_group_sim_bin_for_present.pop('BRA', None)
            remove_uni_country_within_each_political_group_group_sim_bin_for_present.pop('ZAF', None)
            fig = plt.figure(figsize=(10, 8))

            cur_country_idx = 0
            for country in remove_uni_country_within_each_political_group_group_sim_bin_for_present:
                # plt.plot(uni_country_within_each_political_group_group_bin_for_present[group], uni_country_within_each_political_group_group_sim_bin_for_present[group], marker="o")
                plt.plot(uni_country_within_each_political_group_group_bin_for_present[country][:12],
                         remove_uni_country_within_each_political_group_group_sim_bin_for_present[country][:12],
                         color=country_color[cur_country_idx], marker="o")
                plt.errorbar(uni_country_within_each_political_group_group_bin_for_present[country][:12],
                             remove_uni_country_within_each_political_group_group_sim_bin_for_present[country][:12],
                             color=country_color[cur_country_idx],
                             yerr=[
                                 remove_uni_country_within_each_political_group_95interval_error_bin_for_present[country][:12],
                                 remove_uni_country_within_each_political_group_95interval_error_bin_for_present[country][
                                 :12]], fmt='o', markersize=8, capsize=20)

                cur_country_idx += 1
            plt.plot(uni_country_within_each_political_group_ref_bin_for_present[:12],
                     uni_country_within_each_political_group_ref_sim_bin_for_present[:12], color='black', marker="o")
            plt.errorbar(uni_country_within_each_political_group_ref_bin_for_present[:12],
                         uni_country_within_each_political_group_ref_sim_bin_for_present[:12],
                         yerr=[uni_country_within_each_political_group_ref_95interval_error_bin_for_present[:12],
                               uni_country_within_each_political_group_ref_95interval_error_bin_for_present[:12]],
                         color='black', fmt='o', markersize=8, capsize=20)

            # plt.xticks(ticks=range(len(uni_country_within_each_political_group_group_bin_for_present[group])), labels=month_label, fontsize=20)
            plt.xticks(ticks=range(12), labels=['', 'Jan', '', 'Feb', '', 'Mar', '', 'Apr', '', 'May', '', 'Jun'],
                       fontsize=20)
            plt.yticks(size=20)
            plt.xlabel('Time', fontsize=32)
            plt.ylabel('Similarity', fontsize=32)
            cur_legend = []
            for country_alpha3 in list(uni_country_within_each_political_group_group_sim_bin_for_present.keys()):
                cur_legend.append(country_alpha3_geography[country_alpha3]['country_full_name'])
            cur_legend.append(group.upper())
            plt.legend(cur_legend, fontsize=16)

            # plt.show()
            plt.savefig(f"{output_dir}/time_sim_{group}_remove_uni_country_within_each.png")


            # plotting similairty as per countries within multiple political groups, only nato-eunion in this case (no fraction so far or needed)
            uni_country_within_2_political_group_pair = political_group_pair.loc[political_group_pair["main_country1"] == political_group_pair["main_country2"]]
            uni_country_within_2_political_group_pair = uni_country_within_2_political_group_pair.loc[political_group_pair["political_groups_num"] == 2]
            uni_country_within_2_political_group_pair["uni_political_groups"] = uni_country_within_2_political_group_pair.apply(lambda x: x["political_groups"], axis=1)
            uni_country_within_2_political_groups_dict = {}
            uni_country_within_2_political_groups_dict["nato-eunion"] = uni_country_within_2_political_group_pair

            uni_country_within_2_political_groups_group_bin_dict_list = defaultdict(lambda: defaultdict(list))
            uni_country_within_2_political_groups_group_bin_dict = defaultdict(lambda: defaultdict(float))
            uni_country_within_2_political_groups_group_sim_bin_for_present = defaultdict(list)
            uni_country_within_2_political_groups_group_count_bin_for_present = defaultdict(list)
            uni_country_within_2_political_groups_group_bin_for_present = defaultdict(list)

            uni_country_within_2_political_groups_ref_bin_dict_list = defaultdict(list)
            uni_country_within_2_political_groups_ref_sim_bin_for_present = []
            uni_country_within_2_political_groups_ref_count_bin_for_present = []
            uni_country_within_2_political_groups_ref_bin_for_present = []

            uni_country_within_2_political_groups_pair_groups = uni_country_within_2_political_groups_dict["nato-eunion"]["main_country1"].to_list()
            uni_country_within_2_political_groups_pair_bin = uni_country_within_2_political_groups_dict["nato-eunion"]["time_bin"].to_list()
            uni_country_within_2_political_groups_pair_sim = uni_country_within_2_political_groups_dict["nato-eunion"]["similarity"].to_list()
            for i in range(len(uni_country_within_2_political_groups_pair_groups)):
                country = uni_country_within_2_political_groups_pair_groups[i]
                uni_country_within_2_political_groups_group_bin_dict_list[country][
                    uni_country_within_2_political_groups_pair_bin[i]].append(uni_country_within_2_political_groups_pair_sim[i])
                uni_country_within_2_political_groups_ref_bin_dict_list[
                    uni_country_within_2_political_groups_pair_bin[i]].append(uni_country_within_2_political_groups_pair_sim[i])
            for country in uni_country_within_2_political_groups_group_bin_dict_list:
                for bin in uni_country_within_2_political_groups_group_bin_dict_list[country]:
                    uni_country_within_2_political_groups_group_bin_dict[country][bin] = np.mean(uni_country_within_2_political_groups_group_bin_dict_list[country][bin])
            for country in uni_country_within_2_political_groups_group_bin_dict:
                cur_bins = list(uni_country_within_2_political_groups_group_bin_dict_list[country].keys())
                cur_bins.sort()
                for bin in cur_bins:
                    uni_country_within_2_political_groups_group_sim_bin_for_present[country].append(uni_country_within_2_political_groups_group_bin_dict[country][bin])
                    uni_country_within_2_political_groups_group_count_bin_for_present[country].append(np.log(len(uni_country_within_2_political_groups_group_bin_dict_list[country][bin]) /indexes_stats_time_country_dict[country][bin]))
                uni_country_within_2_political_groups_group_bin_for_present[country] = cur_bins
            ref_bins = list(uni_country_within_2_political_groups_ref_bin_dict_list.keys())
            ref_bins.sort()
            for bin in ref_bins:
                uni_country_within_2_political_groups_ref_sim_bin_for_present.append(np.mean(uni_country_within_2_political_groups_ref_bin_dict_list[bin]))
                uni_country_within_2_political_groups_ref_count_bin_for_present.append(np.log(len(uni_country_within_2_political_groups_ref_bin_dict_list[bin]) / indexes_stats_time_country_2_groups_ref_dict['nato-eunion'][bin]))
            uni_country_within_2_political_groups_ref_bin_for_present = ref_bins


            fig = plt.figure(figsize=(10, 8))
            for country in uni_country_within_2_political_groups_group_sim_bin_for_present:
                # plt.plot(uni_country_within_2_political_groups_group_bin_for_present, uni_country_within_2_political_groups_group_sim_bin_for_present, marker="o")
                plt.plot(uni_country_within_2_political_groups_group_bin_for_present[country][:12], uni_country_within_2_political_groups_group_sim_bin_for_present[country][:12], marker="o")
            plt.plot(uni_country_within_2_political_groups_ref_bin_for_present[:12], uni_country_within_2_political_groups_ref_sim_bin_for_present[:12], marker="o")

            # plt.xticks(ticks=range(len(uni_country_within_2_political_groups_group_bin_for_present)), labels=month_label, fontsize=20)
            plt.xticks(ticks=range(12), labels=['', 'Jan', '', 'Feb', '', 'Mar', '', 'Apr', '', 'May', '', 'Jun'], fontsize=20)
            plt.yticks(size=20)
            plt.xlabel('Time', fontsize=32)
            plt.ylabel('Similarity', fontsize=32)
            cur_legend = []
            for country_alpha3 in list(uni_country_within_2_political_groups_group_sim_bin_for_present.keys()):
                cur_legend.append(country_alpha3_geography[country_alpha3]['country_full_name'])
            cur_legend.append('nato-eunion'.upper())
            plt.legend(cur_legend, fontsize=16)

            # plt.show()
            plt.savefig(f"{output_dir}/time_sim_nato-eunion_uni_country_within_2.png")


            fig = plt.figure(figsize=(10, 8))
            for country in uni_country_within_2_political_groups_group_count_bin_for_present:
                # plt.plot(uni_country_within_2_political_groups_group_bin_for_present, uni_country_within_2_political_groups_group_sim_bin_for_present, marker="o")
                plt.plot(uni_country_within_2_political_groups_group_bin_for_present[country][:12], uni_country_within_2_political_groups_group_count_bin_for_present[country][:12], marker="o")
            plt.plot(uni_country_within_2_political_groups_ref_bin_for_present[:12], uni_country_within_2_political_groups_ref_count_bin_for_present[:12], marker="o")

            # plt.xticks(ticks=range(len(uni_country_within_2_political_groups_group_bin_for_present)), labels=month_label, fontsize=20)
            plt.xticks(ticks=range(12), labels=['', 'Jan', '', 'Feb', '', 'Mar', '', 'Apr', '', 'May', '', 'Jun'], fontsize=20)
            plt.yticks(size=20)
            plt.xlabel('Time', fontsize=32)
            plt.ylabel('Fraction', fontsize=32)
            cur_legend = []
            for country_alpha3 in list(uni_country_within_2_political_groups_group_count_bin_for_present.keys()):
                cur_legend.append(country_alpha3_geography[country_alpha3]['country_full_name'])
            cur_legend.append('nato-eunion'.upper())
            plt.legend(cur_legend, fontsize=16)

            # plt.show()
            plt.savefig(f"{output_dir}/time_frac_nato-eunion_uni_country_within_2.png")

    # temporal analysis with diplomatic relation
    if args.option == "dta":
        time_bin = 12

        diplomatic_pair = pairs.copy(deep=True)
        diplomatic_pair = diplomatic_pair.dropna(subset=["longitude1", "longitude2", "latitude1", "latitude2", "country_full_name1", "country_full_name2"])
        diplomatic_pair["time_bin"] = diplomatic_pair.apply( lambda x: round((x['date1'] + x['date2'])/2/time_bin), axis=1)



        # diplomatic relation flow
        country_pair_diplomatic_relation = pd.read_csv("country_info/diplomatic_relation/Diplometrics Diplomatic Representation 1960-2020_20211215.csv")
        country_pair_diplomatic_relation = country_pair_diplomatic_relation[country_pair_diplomatic_relation["Year"] == 2020]
        country_pair_diplomatic_relation = country_pair_diplomatic_relation[["Destination", "Sending Country", "Embassy"]]

        country_pair_diplomatic_relation_dict = {key: country_pair_diplomatic_relation[key].to_list() for key in
                                                 country_pair_diplomatic_relation.keys()}
        country_pair_diplomatic_relation_value_dict = defaultdict(lambda: defaultdict(float))
        for country_idx in range(len(country_pair_diplomatic_relation_dict["Destination"])):
            try:
                cur_destination = country_pair_diplomatic_relation_dict["Destination"][country_idx]
                cur_sending = country_pair_diplomatic_relation_dict["Sending Country"][country_idx]

                country_pair_diplomatic_relation_value_dict[cur_destination][cur_sending] = country_pair_diplomatic_relation_dict["Embassy"][country_idx]
                country_pair_diplomatic_relation_value_dict[cur_sending][cur_destination] = country_pair_diplomatic_relation_dict["Embassy"][country_idx]
            except:
                pass

        diplomatic_pair["diplomatic"] = diplomatic_pair.apply(lambda x: int(country_pair_diplomatic_relation_value_dict[x["country_full_name1"]][x["country_full_name2"]]), axis=1)


        # temporal analysis for news article similarity of political groups
        diplomatic_group_bin_dict_list = defaultdict(lambda: defaultdict(list))
        diplomatic_group_bin_dict = defaultdict(lambda: defaultdict(float))
        diplomatic_group_bin_95interval_error = defaultdict(lambda: defaultdict(float))

        diplomatic_group_sim_bin_for_present = defaultdict(list)
        diplomatic_group_bin_95interval_error_for_present = defaultdict(list)
        diplomatic_group_bin_for_present = defaultdict(list)

        overall_diplomatic_group_bin_dict_list = defaultdict(list)
        overall_diplomatic_group_bin_dict = defaultdict(float)
        overall_diplomatic_group_bin_95interval_error = defaultdict(float)

        overall_diplomatic_group_sim_bin_for_present = []
        overall_diplomatic_group_bin_95interval_error_for_present = []
        overall_diplomatic_group_bin_for_present = []


        intra_country_bin_dict_list = defaultdict(list)
        intra_country_bin_dict = defaultdict(float)
        intra_country_bin_95interval_error = defaultdict(float)

        intra_country_sim_bin_for_present = []
        intra_country_bin_95interval_error_for_present = []
        intra_country_bin_for_present = []


        diplomatic_pair_groups = diplomatic_pair["diplomatic"].to_list()
        diplomatic_pair_bin = diplomatic_pair["time_bin"].to_list()
        diplomatic_pair_sim = diplomatic_pair["similarity"].to_list()
        diplomatic_pair_country_full_name1 = diplomatic_pair["country_full_name1"].to_list()
        diplomatic_pair_country_full_name2 = diplomatic_pair["country_full_name2"].to_list()

        # # trying to have similar pair number for each country/country-pair at a certain timestamp
        # diplomatic_pair_group_bin_country = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

        for i in range(len(diplomatic_pair_groups)):
            cur_diplomatic = diplomatic_pair_groups[i]
            if diplomatic_pair_country_full_name1[i] != diplomatic_pair_country_full_name2[i]:
                diplomatic_group_bin_dict_list[cur_diplomatic][diplomatic_pair_bin[i]].append(diplomatic_pair_sim[i])
            else:
                intra_country_bin_dict_list[diplomatic_pair_bin[i]].append(diplomatic_pair_sim[i])


        for diplomatic in diplomatic_group_bin_dict_list:
            for bin in diplomatic_group_bin_dict_list[diplomatic]:
                diplomatic_group_bin_dict[diplomatic][bin] = np.mean(diplomatic_group_bin_dict_list[diplomatic][bin])

                temp_low, temp_high = st.t.interval(alpha=0.95,
                                                    df=len(diplomatic_group_bin_dict_list[diplomatic][bin]) - 1,
                                                    loc=np.mean(diplomatic_group_bin_dict_list[diplomatic][bin]),
                                                    scale=st.stats.sem(diplomatic_group_bin_dict_list[diplomatic][bin]))
                diplomatic_group_bin_95interval_error[diplomatic][bin] = diplomatic_group_bin_dict[diplomatic][bin] - temp_low

        for diplomatic in diplomatic_group_bin_dict_list:
            for bin in diplomatic_group_bin_dict_list[diplomatic]:
                overall_diplomatic_group_bin_dict_list[bin] = overall_diplomatic_group_bin_dict_list[bin] + diplomatic_group_bin_dict_list[diplomatic][bin]
        for bin in overall_diplomatic_group_bin_dict_list:
            overall_diplomatic_group_bin_dict[bin] = np.mean(overall_diplomatic_group_bin_dict_list[bin])

            temp_low, temp_high = st.t.interval(alpha=0.95,
                                                df=len(overall_diplomatic_group_bin_dict_list[bin]) - 1,
                                                loc=np.mean(overall_diplomatic_group_bin_dict_list[bin]),
                                                scale=st.stats.sem(overall_diplomatic_group_bin_dict_list[bin]))
            overall_diplomatic_group_bin_95interval_error[bin] = overall_diplomatic_group_bin_dict[bin] - temp_low


        for diplomatic in diplomatic_group_bin_dict:
            cur_bins = list(diplomatic_group_bin_dict_list[diplomatic].keys())
            cur_bins.sort()
            for bin in cur_bins:
                diplomatic_group_sim_bin_for_present[diplomatic].append(diplomatic_group_bin_dict[diplomatic][bin])
                diplomatic_group_bin_95interval_error_for_present[diplomatic].append(diplomatic_group_bin_95interval_error[diplomatic][bin])
            diplomatic_group_bin_for_present[diplomatic] = cur_bins

        cur_bins = list(overall_diplomatic_group_bin_dict_list.keys())
        cur_bins.sort()
        for bin in cur_bins:
            overall_diplomatic_group_sim_bin_for_present.append(overall_diplomatic_group_bin_dict[bin])
            overall_diplomatic_group_bin_95interval_error_for_present.append(overall_diplomatic_group_bin_95interval_error[bin])
        overall_diplomatic_group_bin_for_present = cur_bins

        for bin in intra_country_bin_dict_list:
            intra_country_bin_dict[bin] = np.mean(intra_country_bin_dict_list[bin])

            temp_low, temp_high = st.t.interval(alpha=0.95,
                                                df=len(intra_country_bin_dict_list[bin]) - 1,
                                                loc=np.mean(intra_country_bin_dict_list[bin]),
                                                scale=st.stats.sem(intra_country_bin_dict_list[bin]))
            intra_country_bin_95interval_error[bin] = intra_country_bin_dict[bin] - temp_low

        cur_bins = list(diplomatic_group_bin_dict_list[diplomatic].keys())
        cur_bins.sort()
        for bin in cur_bins:
            intra_country_sim_bin_for_present.append(intra_country_bin_dict[bin])
            intra_country_bin_95interval_error_for_present.append(intra_country_bin_95interval_error[bin])
        intra_country_bin_for_present = cur_bins

        diplomatic_group_sim_bin_for_present.pop(0)
        diplomatic_group_sim_bin_for_present.pop(1)
        diplomatic_group_color = {4:"purple", 6:"orange"}
        diplomatic_group_sim_bin_for_present_legend = {}
        for group in diplomatic_group_sim_bin_for_present:
            if group == 4:
                diplomatic_group_sim_bin_for_present_legend["Charge d\' affair"] = group
            if group == 6:
                diplomatic_group_sim_bin_for_present_legend["Ambassador"] = group

        fig = plt.figure(figsize=(10, 8))
        for group in diplomatic_group_sim_bin_for_present:
            # plt.plot(diplomatic_group_bin_for_present[group], diplomatic_group_sim_bin_for_present[group], marker="o")
            plt.plot(diplomatic_group_bin_for_present[group][:12], diplomatic_group_sim_bin_for_present[group][:12], color=diplomatic_group_color[group], marker="o")
            plt.errorbar(diplomatic_group_bin_for_present[group][:12], diplomatic_group_sim_bin_for_present[group][:12], color=diplomatic_group_color[group],
                         yerr=[diplomatic_group_bin_95interval_error_for_present[group][:12],
                               diplomatic_group_bin_95interval_error_for_present[group][:12]], fmt='o',
                         markersize=8, capsize=20)

        # plt.xticks(ticks=range(len(diplomatic_group_bin_for_present[group])), labels=month_label, fontsize=20)
        plt.xticks(ticks=range(12), labels=['', 'Jan', '', 'Feb', '', 'Mar', '', 'Apr', '', 'May', '', 'Jun'], fontsize=20)
        plt.yticks(size=20)
        plt.xlabel('Time', fontsize=24)
        plt.ylabel('Synchronization', fontsize=24)
        cur_legend = list(diplomatic_group_sim_bin_for_present_legend.keys())
        plt.legend(cur_legend, fontsize=20)

        # plt.show()
        plt.savefig(f"{output_dir}/time_sim_diplomatic.png")

        # overall
        fig = plt.figure(figsize=(10, 8))
        plt.plot(overall_diplomatic_group_bin_for_present[:12], overall_diplomatic_group_sim_bin_for_present[:12],color="purple", marker="o")
        plt.errorbar(overall_diplomatic_group_bin_for_present[:12], overall_diplomatic_group_sim_bin_for_present[:12],color="purple",
                     yerr=[overall_diplomatic_group_bin_95interval_error_for_present[:12],
                           overall_diplomatic_group_bin_95interval_error_for_present[:12]], fmt='o',
                     markersize=8, capsize=20)

        # plt.xticks(ticks=range(len(diplomatic_group_bin_for_present[group])), labels=month_label, fontsize=20)
        plt.xticks(ticks=range(12), labels=['', 'Jan', '', 'Feb', '', 'Mar', '', 'Apr', '', 'May', '', 'Jun'], fontsize=20)
        plt.yticks(size=20)
        plt.xlabel('Time', fontsize=24)
        plt.ylabel('Synchronization', fontsize=24)

        # plt.show()
        plt.savefig(f"{output_dir}/time_sim_overall.png")


        fig = plt.figure(figsize=(10, 8))
        plt.plot(intra_country_bin_for_present[:12], intra_country_sim_bin_for_present[:12], color ='green', marker="o")
        plt.errorbar(intra_country_bin_for_present[:12], intra_country_sim_bin_for_present[:12], color ='green',
                     yerr=[intra_country_bin_95interval_error_for_present[:12],
                           intra_country_bin_95interval_error_for_present[:12]], fmt='o',
                     markersize=8, capsize=20)

        # plt.xticks(ticks=range(len(diplomatic_group_bin_for_present[group])), labels=month_label, fontsize=20)
        plt.xticks(ticks=range(12), labels=['', 'Jan', '', 'Feb', '', 'Mar', '', 'Apr', '', 'May', '', 'Jun'], fontsize=20)
        plt.yticks(size=20)
        plt.xlabel('Time', fontsize=24)
        plt.ylabel('Synchronization', fontsize=24)

        # plt.show()
        plt.savefig(f"{output_dir}/time_sim_diplomatic_intra_country.png")






    # geographic distance analysis
    if args.option == "ga":
        distance_pairs = pairs.dropna(subset=["longitude1", "longitude2", "latitude1", "latitude2"])
        distance_pairs["dist"] = distance_pairs.apply(lambda x: geopy.distance.geodesic((x["latitude1"], x["longitude1"]), (x["latitude2"], x["longitude2"])).km, axis=1)
        distance_pairs = distance_pairs[distance_pairs["dist"] != 0]
        distance_pairs["dist_class"] = distance_pairs.apply(lambda x: match_to_distance_class(x["dist"],distance_class), axis=1)

        min_pair_num = 1000

        distance_pairs_group_by_dist_mean = dict(distance_pairs.groupby(["dist_class"])["similarity"].mean())
        distance_pairs_group_by_dist_count = dict(distance_pairs.groupby(["dist_class"])["similarity"].count())

        distance_for_present = []
        distance_sim_for_present = []
        distance_count_for_present = []
        for dist_class in distance_pairs_group_by_dist_mean:
            if distance_pairs_group_by_dist_count[dist_class] >= min_pair_num:
                distance_for_present.append(distance_class[dist_class])
                distance_sim_for_present.append(distance_pairs_group_by_dist_mean[dist_class])
                distance_count_for_present.append(np.log(distance_pairs_group_by_dist_count[dist_class]))

        fig = plt.figure(figsize=(10, 8))
        plt.plot(range(len(distance_for_present)), distance_sim_for_present, marker="o")

        plt.xticks(ticks=range(len(distance_for_present)), labels=distance_for_present, fontsize=20)
        plt.yticks(size=20)
        plt.xlabel('Distance(km)', fontsize=32)
        plt.ylabel('Similarity', fontsize=32)

        # plt.show()
        plt.savefig(f"{output_dir}/dist_sim_inter_country.png")

        fig = plt.figure(figsize=(10, 8))
        plt.plot(range(len(distance_for_present)), distance_count_for_present, marker="o")
        plt.xticks(ticks=range(len(distance_for_present)), labels=distance_for_present, fontsize=20)
        plt.yticks(size=20)
        plt.xlabel('Distance(km)', fontsize=32)
        plt.ylabel('Pair Count', fontsize=32)

        # plt.show()
        plt.savefig(f"{output_dir}/dist_count_inter_country.png")

    # regression model to evaluate importance of different factors on the inter-country level
    # args.cluster_sim can't be used at the same time as args.opt == "rm" or args.opt == "rm_intra"
    if args.option == "rm":
        filtered_out = 1
        # filtered_out = 0

        # subsampling = 1
        subsampling = 0
        subsampling_num = 200

        cluster_validation = 1
        # cluster_validation = 0

        # search_metric = "vif"
        # search_metric = "aic"

        time_bin = 180
        # time_bin = 60
        # time_bin = 12

        min_pair_count = 20
        top_k_param = 10

        country_cls_df = pd.read_csv("network_data/country_cluster/country_cluster.csv")
        country_cls_dict = country_cls_df.to_dict()
        country_cls_list = defaultdict(list)
        for i in range(country_cls_df.shape[0]):
            for cls in country_cls_dict:
                if cls == "Unnamed: 0":
                    alpha3 = country_cls_dict[cls][i]
                else:
                    country_cls_list[alpha3].append(country_cls_dict[cls][i])
        for alpha3 in country_cls_list:
            print(sum(country_cls_list[alpha3]))

        for cur_bin in range(1, int(180/time_bin)+1):
            print("***********************")
            print("***********************")
            print("cur bin is ", cur_bin)
            print("***********************")
            print("***********************")

            rm_pairs = pairs.dropna(subset=["longitude1", "longitude2", "latitude1", "latitude2"])
            rm_pairs = rm_pairs.loc[rm_pairs["main_country1"] != rm_pairs["main_country2"]]
            rm_pairs["time_bin"] = rm_pairs.apply(lambda x: round((x['date1'] + x['date2']) / 2 / time_bin), axis=1)
            if time_bin != 180:
                rm_pairs = rm_pairs[rm_pairs["time_bin"] == cur_bin]
            # rm_pairs = rm_pairs[rm_pairs["time_bin"] == cur_bin]

            rm_pairs["same_lang"] = rm_pairs.apply(lambda x: 1 if x['language1'] == x['a2_language'] else 0, axis=1)
            rm_pairs["same_spec_lang"] = rm_pairs.apply(lambda x: lang_list.index(x['language1']) if x['language1'] == x['a2_language'] else -1, axis=1)

            # filtering the intra-country pairs whose country official language in not covered in the 10 languages of our annotation
            country_official_languages = pd.read_csv("country_info/country_official_language.csv")
            country_official_languages["Country"] = country_official_languages.apply(lambda x: x['Country'].replace("\xa0", ""), axis=1)
            country_official_languages["Official language"] = country_official_languages.apply(lambda x: x['Official language'].replace("\xa0", "").replace("\u2028", "\n").split("\n"), axis=1)
            country_official_languages["Official language family"] = country_official_languages.apply(lambda x: [LANG_FAMILY[LANG_FULL_NAME_MAP[lang]] for lang in x['Official language'] if lang in LANG_FULL_NAME_MAP], axis=1)

            country_official_languages = pd.merge(country_geography_list, country_official_languages, how='left',on='Country')

            '''load country neighbor/border info'''
            country_neighbors = pd.read_csv("country_info/country_neighbors.csv")
            country_neighbors["name"] = country_neighbors.apply(lambda x: x["name"].split(",")[0], axis=1)
            country_neighbors["borders"] = country_neighbors.apply(lambda x: x["borders"].split(",") if isinstance(x["borders"],str) else [], axis=1)
            country_neighbors = country_neighbors.drop(["status", "currencies", "capital", "region", "subregion", "languages", "latlng", "area", "demonyms"],axis=1)
            country_neighbors.rename(columns={'name': 'Country'}, inplace=True)
            country_neighbors = pd.merge(country_official_languages, country_neighbors, how='left',on='Country')

            '''load continent info'''
            country_continent = pd.read_csv("country_info/country_continent.csv")
            country_continent = pd.merge(country_neighbors, country_continent, how='left', on='Country')

            '''load democracy index info'''
            country_democracy_index_list = pd.read_csv("bias_dataset/2019_democracy_index/2019_democracy_index.csv")
            country_democracy_index_list = country_democracy_index_list[country_democracy_index_list["year"] == 2019]
            country_democracy_index_list.rename(columns={'country': 'Country'}, inplace=True)
            country_democracy_index_list = pd.merge(country_continent, country_democracy_index_list, how='left', on='Country')

            '''load press freedom index info'''
            country_press_freedom_index_list = pd.read_csv("country_info/country_press_freedom_index.csv")
            country_press_freedom_index_list = pd.merge(country_democracy_index_list, country_press_freedom_index_list, how='left', on='Country')

            '''load unitary state info'''
            country_unitary_state_list = pd.read_csv("country_info/country_unitary_state.csv")
            country_unitary_state_list = pd.merge(country_press_freedom_index_list, country_unitary_state_list, how='left', on='Country')

            '''load gdp info'''
            country_gdp_list = pd.read_csv("country_info/country_gdp.csv")
            country_gdp_list = country_gdp_list[["Country Name", "2020"]]
            country_gdp_list.rename(columns={"Country Name": 'Country'}, inplace=True)
            country_gdp_list.rename(columns={"2020": '2020_gdp'}, inplace=True)

            country_gdp_list = pd.merge(country_unitary_state_list, country_gdp_list, how='left', on='Country')

            '''load gini index info'''
            country_gini_index_list = pd.read_csv("country_info/country_gini_index.csv")
            country_gini_index_list = pd.merge(country_gdp_list, country_gini_index_list, how='left', on='Alpha-3 code')

            '''load peace index info'''
            country_peace_index_list = pd.read_csv("country_info/country_peace_index.csv")
            country_peace_index_list = country_peace_index_list[["Country", "2020 Rate"]]
            country_peace_index_list.rename(columns={"2020 Rate": 'peace_index'}, inplace=True)
            # to align the sign with other factors
            country_peace_index_list['Country'] = country_peace_index_list.apply(lambda x: x['Country'].replace("\xa0",""), axis=1)
            country_peace_index_list['peace_index'] = country_peace_index_list.apply(lambda x: -x['peace_index'], axis=1)
            country_peace_index_list = pd.merge(country_gini_index_list, country_peace_index_list, how='left', on='Country')

            '''load political group info'''
            # dict for speedy query
            # one country or country pair can belong to multiple political group
            country_syn_list = country_gini_index_list

            political_group_df_dict = {}
            political_group_alpha3_dict = {}
            for group in political_group_list:
                political_group_df_dict[group] = pd.read_csv(f"country_info/political_group/{group}.csv")
                # nato csv contains some other countries
                if group == "nato":
                    political_group_df_dict["nato"] = political_group_df_dict["nato"][political_group_df_dict["nato"]["Category"] == "NATO"]

                political_group_df_dict[group].rename(columns={'Alpha-3': 'Alpha-3 code'}, inplace=True)
                political_group_df_dict[group] = political_group_df_dict[group][['Alpha-3 code']]
                political_group_df_dict[group] = political_group_df_dict[group].drop_duplicates()
                political_group_alpha3_dict[group] = {alpha3: 1 for alpha3 in political_group_df_dict[group]['Alpha-3 code'].to_list()}

                political_group_df_dict[group][group] = political_group_df_dict[group].apply(lambda x:group, axis=1)
                country_syn_list = pd.merge(country_syn_list, political_group_df_dict[group], how='left', on='Alpha-3 code')

            country_syn_list = country_syn_list.dropna(subset=["eiu", "Press Freedom", '2020_gdp', "gini_index"])

            '''aggregate article pairs as per each country pair'''
            rm_pairs_dict = {key: rm_pairs[key].to_list() for key in rm_pairs.keys()}
            rm_edge_sim = defaultdict(lambda: defaultdict(list))  # edge's size (similarity)
            rm_edge_same_lang = defaultdict(lambda: defaultdict(list))
            rm_edge_same_spec_lang = defaultdict(lambda: defaultdict(list))
            for i in range(len(rm_pairs)):
                cur_country1 = rm_pairs_dict["main_country1"][i]
                cur_country2 = rm_pairs_dict["main_country2"][i]

                same_lang_vec = [0 for i in range(len(lang_list))]
                if rm_pairs_dict["same_spec_lang"][i] >= 0:
                    same_lang_vec[rm_pairs_dict["same_spec_lang"][i]] = 1

                if cur_country1 < cur_country2:
                    rm_edge_sim[cur_country1][cur_country2].append(rm_pairs_dict["similarity"][i])
                    rm_edge_same_lang[cur_country1][cur_country2].append(rm_pairs_dict["same_lang"][i])
                    rm_edge_same_spec_lang[cur_country1][cur_country2].append(same_lang_vec)
                else:
                    rm_edge_sim[cur_country2][cur_country1].append(rm_pairs_dict["similarity"][i])
                    rm_edge_same_lang[cur_country2][cur_country1].append(rm_pairs_dict["same_lang"][i])
                    rm_edge_same_spec_lang[cur_country2][cur_country1].append(same_lang_vec)


            country_pair_syn_dict_dict_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
            country_syn_dict = {key: country_syn_list[key].to_list() for key in country_syn_list.keys()}
            check_lang_country = []
            for border_idx in range(len(country_syn_dict['borders'])):
                if not isinstance(country_syn_dict['borders'][border_idx], list):
                    country_syn_dict['borders'][border_idx] = []
            country_to_alpha3_dict = {}
            alpha2_to_alpha3_dict = {}
            for i in range(len(country_syn_dict["Country"])):
                country_to_alpha3_dict[country_syn_dict["Country"][i]] = country_syn_dict["Alpha-3 code"][i]
                alpha2_to_alpha3_dict[country_syn_dict["Alpha-2 code"][i]] = country_syn_dict["Alpha-3 code"][i]

            # diplomatic relation flow
            country_pair_diplomatic_relation = pd.read_csv("country_info/diplomatic_relation/Diplometrics Diplomatic Representation 1960-2020_20211215.csv")
            country_pair_diplomatic_relation = country_pair_diplomatic_relation[country_pair_diplomatic_relation["Year"] == 2020]
            country_pair_diplomatic_relation = country_pair_diplomatic_relation[["Destination", "Sending Country", "Embassy"]]

            country_pair_diplomatic_relation_dict = {key: country_pair_diplomatic_relation[key].to_list() for key in country_pair_diplomatic_relation.keys()}
            country_pair_diplomatic_relation_value_dict = defaultdict(lambda: defaultdict(float))
            for country_idx in range(len(country_pair_diplomatic_relation_dict["Destination"])):
                try:
                    cur_destination_alpha3 = country_to_alpha3_dict[country_pair_diplomatic_relation_dict["Destination"][country_idx]]
                    cur_sending_alpha3 = country_to_alpha3_dict[country_pair_diplomatic_relation_dict["Sending Country"][country_idx]]

                    country_pair_diplomatic_relation_value_dict[cur_destination_alpha3][cur_sending_alpha3] = country_pair_diplomatic_relation_dict["Embassy"][country_idx]
                    country_pair_diplomatic_relation_value_dict[cur_sending_alpha3][cur_destination_alpha3] = country_pair_diplomatic_relation_dict["Embassy"][country_idx]
                except:
                    pass
            # investment flow
            country_pair_investment = pd.read_csv("country_info/economy_flow_raw_data/country_investment_flow.csv")
            country_pair_investment = country_pair_investment[country_pair_investment["INDICATOR"] == "VALUE"]
            country_pair_investment = country_pair_investment[["REPORT_CTRY", "PARTNER_CTRY", "2018"]]

            country_pair_investment_dict = {key: country_pair_investment[key].to_list() for key in country_pair_investment.keys()}
            country_pair_investment_value_dict = defaultdict(lambda: defaultdict(float))
            for country_idx in range(len(country_pair_investment_dict["REPORT_CTRY"])):
                try:
                    cur_destination_alpha3 = alpha2_to_alpha3_dict[country_pair_investment_dict["REPORT_CTRY"][country_idx]]
                    cur_sending_alpha3 = alpha2_to_alpha3_dict[country_pair_investment_dict["PARTNER_CTRY"][country_idx]]

                    country_pair_investment_value_dict[cur_destination_alpha3][cur_sending_alpha3] += country_pair_investment_dict["2018"][country_idx]
                    country_pair_investment_value_dict[cur_sending_alpha3][cur_destination_alpha3] += country_pair_investment_dict["2018"][country_idx]
                except:
                    pass
            # trade flow
            country_pair_trade = pd.read_csv("country_info/economy_flow_raw_data/trade_export_flow.csv")
            country_pair_trade = country_pair_trade[["Country Name", "Counterpart Country Name", "Value"]]

            country_pair_trade_dict = {key: country_pair_trade[key].to_list() for key in country_pair_trade.keys()}
            country_pair_trade_value_dict = defaultdict(lambda: defaultdict(float))
            for country_idx in range(len(country_pair_trade["Country Name"])):
                try:
                    cur_destination_alpha3 = country_to_alpha3_dict[country_pair_trade_dict["Country Name"][country_idx]]
                    cur_sending_alpha3 = country_to_alpha3_dict[country_pair_trade_dict["Counterpart Country Name"][country_idx]]

                    country_pair_trade_value_dict[cur_destination_alpha3][cur_sending_alpha3] += country_pair_trade_dict["Value"][country_idx]
                    country_pair_trade_value_dict[cur_sending_alpha3][cur_destination_alpha3] += country_pair_trade_dict["Value"][country_idx]
                except:
                    pass

            # immgration flow
            country_pair_immgration = pd.read_csv("country_info/migration/bilateral_migrationmatrix_2018.csv")

            country_pair_immgration_dict = {key: country_pair_immgration[key].to_list() for key in country_pair_immgration.keys()}
            country_pair_immgration_value_dict = defaultdict(lambda: defaultdict(float))
            for country1_idx in range(len(country_pair_immgration["Country"])):
                for country2_idx in range(len(country_pair_immgration["Country"])):
                    try:
                        cur_country1 = country_pair_immgration_dict["Country"][country1_idx]
                        cur_country2 = country_pair_immgration_dict["Country"][country2_idx]

                        country_pair_immgration_value_dict[cur_country1][cur_country2] += float(country_pair_immgration_dict[cur_country1][cur_country2])
                        country_pair_immgration_value_dict[cur_country2][cur_country1] += float(country_pair_immgration_dict[cur_country1][cur_country2])
                    except:
                        pass


            for i in range(len(country_syn_list)):
                for j in range(len(country_syn_list)):
                    if i == j:
                        continue
                    else:
                        if country_syn_dict["Alpha-3 code"][i] < country_syn_dict["Alpha-3 code"][j]:
                            cur_idx1 = i
                            cur_idx2 = j
                        else:
                            cur_idx1 = j
                            cur_idx2 = i
                        cur_alpha3_1 = country_syn_dict["Alpha-3 code"][cur_idx1]
                        cur_alpha3_2 = country_syn_dict["Alpha-3 code"][cur_idx2]

                        if (cur_alpha3_1 not in rm_edge_sim) or (cur_alpha3_2 not in rm_edge_sim[cur_alpha3_1]):
                            continue


                        if filtered_out:
                            country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["pair_count"] = len(rm_edge_sim[cur_alpha3_1][cur_alpha3_2])
                            country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["avg_sim"] = np.mean(rm_edge_sim[cur_alpha3_1][cur_alpha3_2])

                            country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["sim"] = rm_edge_sim[cur_alpha3_1][cur_alpha3_2]
                            country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["same_lang"] = rm_edge_same_lang[cur_alpha3_1][cur_alpha3_2]
                            country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["same_spec_lang"] = rm_edge_same_spec_lang[cur_alpha3_1][cur_alpha3_2]

                        else:
                            # load filter-out article pairs data
                            geo_index_art_data = defaultdict(int)
                            for pair in indexes_stats.itertuples():
                                cur_main_country = pair[8]
                                cur_art_count = pair[4]
                                geo_index_art_data[cur_main_country] += cur_art_count

                            country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["pair_count"] = geo_index_art_data[cur_alpha3_1] * geo_index_art_data[cur_alpha3_2]
                            country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["avg_sim"] = np.mean(rm_edge_sim[cur_alpha3_1][cur_alpha3_2]) * len(rm_edge_sim[cur_alpha3_1][cur_alpha3_2]) / country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["pair_count"]

                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["Country1"] = country_syn_dict["Country"][cur_idx1]
                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["Country2"] = country_syn_dict["Country"][cur_idx2]

                        if (cur_alpha3_1 in country_syn_dict["borders"][cur_idx2]) or (cur_alpha3_2 in country_syn_dict["borders"][cur_idx1]):
                            country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]['border'] = 1
                        else:
                            country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]['border'] = 0

                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]['dist'] = geopy.distance.geodesic((country_syn_dict["Latitude (average)"][cur_idx1], country_syn_dict["Longitude (average)"][cur_idx1]), (country_syn_dict["Latitude (average)"][cur_idx2], country_syn_dict["Longitude (average)"][cur_idx2])).km

                        # continent similarity
                        if country_syn_dict["Continent"][cur_idx1] == country_syn_dict["Continent"][cur_idx2]:
                            country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]['continent_sim'] = 1
                        else:
                            country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]['continent_sim'] = 0

                        # language categories
                        # About 12 categorical values = 10 languages (or less, because some languages like Polish are official in only one country) + “Not the same language but same family” + “Not the same language and different family” categories
                        try:
                            common_official_langs = list(set(country_syn_dict["Official language"][cur_idx1]).intersection(set(country_syn_dict["Official language"][cur_idx2])))
                        except:
                            common_official_langs = []
                        try:
                            common_official_langs_family = list(set(country_syn_dict["Official language family"][cur_idx1]).intersection(set(country_syn_dict["Official language family"][cur_idx2])))
                        except:
                            common_official_langs_family = []

                        simple_lang_vector = [0 for i in range(1)]
                        if common_official_langs != []:
                            simple_lang_vector[0] = 1
                        # elif common_official_langs_family != []:
                        #     simple_lang_vector[1] = 1
                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["simple_lang_vec"] = simple_lang_vector


                        lang_vector = [0 for i in range(10)]
                        if common_official_langs != []:
                            lang_full_name_map_keys = list(LANG_FULL_NAME_MAP.keys())
                            for lang in common_official_langs:
                                if lang in lang_full_name_map_keys:
                                    lang_vector[lang_full_name_map_keys.index(lang)] = 1
                                    # # check which country speak russian
                                    # if (lang == "Russian"):
                                    #     check_lang_country.append([cur_alpha3_1,cur_alpha3_2,country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["avg_sim"]])

                        # elif common_official_langs_family != []:
                        #     lang_vector[10] = 1
                        # else:
                        #     lang_vector[11] = 1
                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["lang_vec"] = lang_vector

                        # political group categories, 4 kinds
                        common_group = []
                        for group in ['nato','eunion','brics']:
                            if country_syn_dict[group][cur_idx1] == country_syn_dict[group][cur_idx2]:
                                common_group.append(group)

                        across_group = []
                        if (country_syn_dict['nato'][cur_idx1] == 'nato' and country_syn_dict['eunion'][cur_idx2] == 'eunion') or (country_syn_dict['eunion'][cur_idx1] == 'eunion' and country_syn_dict['nato'][cur_idx2] == 'nato'):
                            across_group.append('nato-eunion')
                        if (country_syn_dict['nato'][cur_idx1] == 'nato' and country_syn_dict['brics'][cur_idx2] == 'brics') or (country_syn_dict['brics'][cur_idx1] == 'brics' and country_syn_dict['nato'][cur_idx2] == 'nato'):
                            across_group.append('nato-brics')
                        if (country_syn_dict['eunion'][cur_idx1] == 'eunion' and country_syn_dict['brics'][cur_idx2] == 'brics') or (country_syn_dict['brics'][cur_idx1] == 'brics' and country_syn_dict['eunion'][cur_idx2] == 'eunion'):
                            across_group.append('eunion-brics')

                        simple_political_group_vector = [0]
                        if common_group != []:
                            simple_political_group_vector[0] = 1
                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["simple_political_group_vec"] = simple_political_group_vector



                        # '''disentangled political group featrues'''
                        # political_group_vector = [0 for i in range(4)]
                        # if ('nato' in common_group) and (len(common_group) == 1):
                        #     political_group_vector[0] = 1
                        # elif ('eunion' in common_group) and (len(common_group) == 1):
                        #     political_group_vector[1] = 1
                        # elif ('brics' in common_group) and (len(common_group) == 1):
                        #     political_group_vector[2] = 1
                        # elif ('nato' in common_group) and ('eunion' in common_group) and (len(common_group) == 2):
                        #     political_group_vector[3] = 1

                        '''political group featrues'''
                        political_group_vector = [0 for i in range(3)]
                        if 'nato' in common_group:
                            political_group_vector[0] = 1
                        elif 'eunion' in common_group:
                            political_group_vector[1] = 1
                        elif 'brics' in common_group:
                            political_group_vector[2] = 1

                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["political_group_vec"] = political_group_vector

                        across_group_vector = [0 for i in range(3)]
                        if 'nato-eunion' in across_group:
                            across_group_vector[0] = 1
                        if 'nato-brics' in across_group:
                            across_group_vector[1] = 1
                        if 'eunion-brics' in across_group:
                            across_group_vector[2] = 1

                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["across_group_vec"] = across_group_vector



                        # unitary type combination categories, 6 kinds
                        simple_unitary_vector = [0]
                        if country_syn_dict["Unitary"][cur_idx1] == country_syn_dict["Unitary"][cur_idx2]:
                            simple_unitary_vector[0] = 1
                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["simple_unitary_vector"] = simple_unitary_vector

                        unitary_vector = [0 for i in range(6)]
                        combo_unitary_types = [country_syn_dict["Unitary"][cur_idx1], country_syn_dict["Unitary"][cur_idx2]]
                        if combo_unitary_types == ["unitary_republics", "unitary_republics"]:
                            unitary_vector[0] = 1
                        elif combo_unitary_types == ["federalism", "federalism"]:
                            unitary_vector[1] = 1
                        # elif ("unitary_republics" not in combo_unitary_types) and ("federalism" not in combo_unitary_types):
                        #     unitary_vector[2] = 1
                        elif ("unitary_republics" in combo_unitary_types) and ("federalism" in combo_unitary_types):
                            unitary_vector[3] = 1
                        elif ("unitary_republics" in combo_unitary_types) and ("federalism" not in combo_unitary_types):
                            unitary_vector[4] = 1
                        elif ("unitary_republics" not in combo_unitary_types) and ("federalism" in combo_unitary_types):
                            unitary_vector[5] = 1
                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["unitary_vec"] = unitary_vector

                        # type1: democracy index alignment and press freedom index alignment
                        # if not np.isnan(country_syn_dict["eiu"][cur_idx1]):
                        #     cur_dem_alignment1 = math.ceil(country_syn_dict["eiu"][cur_idx1]/0.5)
                        # else:
                        #     cur_dem_alignment1 = 0
                        # if not np.isnan(country_syn_dict["eiu"][cur_idx2]):
                        #     cur_dem_alignment2 = math.ceil(country_syn_dict["eiu"][cur_idx2]/0.5)
                        # else:
                        #     cur_dem_alignment2 = 0
                        #
                        # if not np.isnan(country_syn_dict["Press Freedom"][cur_idx1]):
                        #     cur_pf_alignment1 = math.ceil(country_syn_dict["Press Freedom"][cur_idx1]/2)
                        # else:
                        #     cur_pf_alignment1 = 0
                        # if not np.isnan(country_syn_dict["Press Freedom"][cur_idx2]):
                        #     cur_pf_alignment2 = math.ceil(country_syn_dict["Press Freedom"][cur_idx2]/2)
                        # else:
                        #     cur_pf_alignment2 = 0
                        #
                        # if cur_dem_alignment1 == cur_dem_alignment2:
                        #     dem_idx_converg = cur_dem_alignment1
                        # else:
                        #     dem_idx_converg = 0
                        # if cur_pf_alignment1 == cur_pf_alignment2:
                        #     pf_idx_converg = cur_pf_alignment1
                        # else:
                        #     pf_idx_converg = 0

                        # type2: democracy index alignment and press freedom index alignment
                        # if not np.isnan(country_syn_dict["eiu"][cur_idx1]) and not np.isnan(country_syn_dict["eiu"][cur_idx2]):
                        #     if abs(country_syn_dict["eiu"][cur_idx1] - country_syn_dict["eiu"][cur_idx2]) <= 0.5:
                        #         dem_idx_converg = (country_syn_dict["eiu"][cur_idx1] + country_syn_dict["eiu"][cur_idx2])/2
                        #     else:
                        #         dem_idx_converg = 0
                        # else:
                        #     dem_idx_converg = 0
                        #
                        # if not np.isnan(country_syn_dict["Press Freedom"][cur_idx1]) and not np.isnan(country_syn_dict["Press Freedom"][cur_idx2]):
                        #     if abs(country_syn_dict["Press Freedom"][cur_idx1] - country_syn_dict["Press Freedom"][cur_idx2]) <= 0.5:
                        #         pf_idx_converg = (country_syn_dict["Press Freedom"][cur_idx1] + country_syn_dict["Press Freedom"][cur_idx2])/2
                        #     else:
                        #         pf_idx_converg = 0
                        # else:
                        #     pf_idx_converg = 0

                        gdp_vec = [0 for i in range(3)]  # 'low-low','high-high','low-high'
                        if not np.isnan(country_syn_dict["2020_gdp"][cur_idx1]) and not np.isnan(country_syn_dict["2020_gdp"][cur_idx2]):
                            # 0 stands for low, 1 stands for high
                            if country_syn_dict["2020_gdp"][cur_idx1] < 500000 * 1000000:
                                gdp_idx_sign1 = 0
                            else:
                                gdp_idx_sign1 = 1
                            if country_syn_dict["2020_gdp"][cur_idx2] < 500000 * 1000000:
                                gdp_idx_sign2 = 0
                            else:
                                gdp_idx_sign2 = 1

                            if gdp_idx_sign1 == gdp_idx_sign2:
                                gdp_converg = 1

                                if gdp_idx_sign1 == 0:
                                    # take this class as reference
                                    # gdp_vec[0] = 1
                                    pass
                                else:
                                    gdp_vec[1] = 1
                            else:
                                gdp_converg = 0
                                gdp_vec[2] = 1

                        else:
                            gdp_converg = float("nan")
                            gdp_vec = [float("nan") for i in range(3)]

                        gini_index_vec = [0 for i in range(3)]  # 'low-low','high-high','low-high'
                        if not np.isnan(country_syn_dict["gini_index"][cur_idx1]) and not np.isnan(country_syn_dict["gini_index"][cur_idx2]):
                            # 0 stands for low, 1 stands for high
                            if country_syn_dict["gini_index"][cur_idx1] < 35:
                                gini_index_idx_sign1 = 0
                            else:
                                gini_index_idx_sign1 = 1
                            if country_syn_dict["gini_index"][cur_idx2] < 35:
                                gini_index_idx_sign2 = 0
                            else:
                                gini_index_idx_sign2 = 1

                            if gini_index_idx_sign1 == gini_index_idx_sign2:
                                gini_index_converg = 1

                                if gini_index_idx_sign1 == 0:
                                    # take this class as reference
                                    # gini_index_vec[0] = 1
                                    pass
                                else:
                                    gini_index_vec[1] = 1
                            else:
                                gini_index_converg = 0
                                gini_index_vec[2] = 1

                        else:
                            gini_index_converg = float("nan")
                            gini_index_vec = [float("nan") for i in range(3)]

                        dem_idx_vec = [0 for i in range(3)] # 'low-low','high-high','low-high'
                        if not np.isnan(country_syn_dict["eiu"][cur_idx1]) and not np.isnan(country_syn_dict["eiu"][cur_idx2]):
                            # 0 stands for low, 1 stands for high
                            if country_syn_dict["eiu"][cur_idx1] < 5:
                                dem_idx_sign1 = 0
                            else:
                                dem_idx_sign1 = 1
                            if country_syn_dict["eiu"][cur_idx2] < 5:
                                dem_idx_sign2 = 0
                            else:
                                dem_idx_sign2 = 1

                            if dem_idx_sign1 == dem_idx_sign2:
                                dem_idx_converg = 1

                                if dem_idx_sign1 == 0:
                                    # take this class as reference
                                    # dem_idx_vec[0] = 1
                                    pass
                                else:
                                    dem_idx_vec[1] = 1
                            else:
                                dem_idx_converg = 0
                                dem_idx_vec[2] = 1

                        else:
                            dem_idx_converg = float("nan")
                            dem_idx_vec = [float("nan") for i in range(3)]

                        # pf_idx_vec = [0 for i in range(3)]  # 'low-low','high-high','low-high'
                        # if not np.isnan(country_syn_dict["Press Freedom"][cur_idx1]) and not np.isnan(country_syn_dict["Press Freedom"][cur_idx2]):
                        #     # 0 stands for low, 1 stands for high
                        #     if country_syn_dict["Press Freedom"][cur_idx1] < 50:
                        #         pf_idx_sign1 = 0
                        #     else:
                        #         pf_idx_sign1 = 1
                        #     if country_syn_dict["Press Freedom"][cur_idx2] < 50:
                        #         pf_idx_sign2 = 0
                        #     else:
                        #         pf_idx_sign2 = 1
                        #
                        #     if pf_idx_sign1 == pf_idx_sign2:
                        #         pf_idx_converg = 1
                        #
                        #         if pf_idx_sign1 == 0:
                        #             pf_idx_vec[0] = 1
                        #         else:
                        #             pf_idx_vec[1] = 1
                        #     else:
                        #         pf_idx_converg = 0
                        #         pf_idx_vec[2] = 1
                        #
                        # else:
                        #     pf_idx_converg = float("nan")
                        #     pf_idx_vec = [float("nan") for i in range(3)]

                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["gdp_converg"] = gdp_converg
                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["gdp_vec"] = gdp_vec

                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["gini_index_converg"] = gini_index_converg
                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["gini_index_vec"] = gini_index_vec

                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["dem_idx_converg"] = dem_idx_converg
                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["dem_idx_vec"] = dem_idx_vec

                        # country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["pf_idx_converg"] = pf_idx_converg
                        # country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["pf_idx_vec"] = pf_idx_vec

                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]['country1'] = country_alpha3_geography[cur_alpha3_1]["country_full_name"]
                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]['country2'] = country_alpha3_geography[cur_alpha3_2]["country_full_name"]

                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["diplomatic"] = country_pair_diplomatic_relation_value_dict[cur_alpha3_1][cur_alpha3_2]
                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["investment"] = country_pair_investment_value_dict[cur_alpha3_1][cur_alpha3_2]
                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["trade"] = country_pair_trade_value_dict[cur_alpha3_1][cur_alpha3_2]
                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["immgration"] = country_pair_immgration_value_dict[cur_alpha3_1][cur_alpha3_2]

            with open("indexes/rm_data_inter_country.json", 'w') as f:
                json.dump(country_pair_syn_dict_dict_dict, f)

            jensenshannon_dict = defaultdict(dict)
            for cur_alpha3_1 in country_pair_syn_dict_dict_dict:
                for cur_alpha3_2 in country_pair_syn_dict_dict_dict[cur_alpha3_1]:
                    try:
                        jensenshannon_dict[cur_alpha3_1][cur_alpha3_2] = distance.jensenshannon(country_cls_list[cur_alpha3_1], country_cls_list[cur_alpha3_2])
                    except:
                        pass

            simple_inter_country_train_data = []
            inter_country_train_data = []
            inter_country_train_label = []
            fitlered_country_pair_num = 0
            for cur_alpha3_1 in country_pair_syn_dict_dict_dict:
                for cur_alpha3_2 in country_pair_syn_dict_dict_dict[cur_alpha3_1]:
                    cur_country_pair = copy.deepcopy(country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2])
                    if cur_country_pair['pair_count'] < min_pair_count:
                        continue

                    try:
                        if cluster_validation:
                            inter_country_train_label.append(jensenshannon_dict[cur_alpha3_1][cur_alpha3_2])
                        else:
                            # Similarity
                            inter_country_train_label.append(cur_country_pair['avg_sim'])
                            # # similarity
                            # inter_country_train_label.append(4 - cur_country_pair['avg_sim'])

                        # simple_inter_country_train_data.append([cur_country_pair['dist']] + cur_country_pair['simple_lang_vec'] + cur_country_pair['simple_political_group_vec'] + [cur_country_pair['dem_idx_diff']] + [cur_country_pair['pf_idx_diff']])
                        simple_inter_country_train_data.append([cur_country_pair['dist']] + cur_country_pair['simple_lang_vec'] + [cur_country_pair['dem_idx_converg']])
                        # inter_country_train_data.append([cur_country_pair['dist']] + [cur_country_pair['continent_sim']] + cur_country_pair['lang_vec'] + cur_country_pair['political_group_vec'] + [cur_country_pair['dem_idx_diff']] + [cur_country_pair['pf_idx_diff']])

                        inter_country_train_data.append([cur_country_pair['dist']] + [cur_country_pair['continent_sim']] + cur_country_pair['lang_vec'] + [cur_country_pair['dem_idx_converg']])

                    except:
                        fitlered_country_pair_num += 1
            print("fitlered_country_pair_num: ", fitlered_country_pair_num)
            # normalization
            minmax = MinMaxScaler()
            simple_inter_country_train_data = minmax.fit_transform(simple_inter_country_train_data)
            inter_country_train_data = minmax.fit_transform(inter_country_train_data)
            print("country pairs total number:", len(inter_country_train_label))

            # model = LinearRegression()
            # model.fit(inter_country_train_data, inter_country_train_label)
            # print("parameter w={},b={}".format(model.coef_, model.intercept_))
            # model_coef_abs = [abs(coef) for coef in model.coef_]
            # top_k_param_idx = heapq.nlargest(top_k_param, range(len(model_coef_abs)), model_coef_abs.__getitem__)
            # top_k_param_weight = [model.coef_[k] for k in top_k_param_idx]

            print("-----------------------------")
            print("-----------------------------")
            print("simple model")
            print("-----------------------------")
            print("-----------------------------")

            print(len(inter_country_train_label))
            print(len(sm.add_constant(simple_inter_country_train_data)))

            model = sm.OLS(inter_country_train_label, sm.add_constant(simple_inter_country_train_data)).fit()
            print("simple model -- parameter with intercept:", model.params)
            model_coef_abs = [abs(coef) for coef in model.params]
            top_k_param_idx = heapq.nlargest(top_k_param, range(len(model_coef_abs)), model_coef_abs.__getitem__)
            top_k_param_weight = [model.params[k] for k in top_k_param_idx]


            print(f"simple model -- the index of top {top_k_param} param (abs):", top_k_param_idx)
            print(f"simple model -- the weight of top {top_k_param} param:", top_k_param_weight)

            print("simple model -- model summary is")
            print()
            print(model.summary())


            print("-----------------------------")
            print("-----------------------------")
            print("complex model")
            print("-----------------------------")
            print("-----------------------------")

            model = sm.OLS(inter_country_train_label, sm.add_constant(inter_country_train_data)).fit()
            print("complex model -- parameter with intercept:", model.params)
            model_coef_abs = [abs(coef) for coef in model.params]
            top_k_param_idx = heapq.nlargest(top_k_param, range(len(model_coef_abs)), model_coef_abs.__getitem__)
            top_k_param_weight = [model.params[k] for k in top_k_param_idx]


            print(f"complex model -- the index of top {top_k_param} param (abs):", top_k_param_idx)
            print(f"complex model -- the weight of top {top_k_param} param:", top_k_param_weight)
            print(f"complex model -- AIC:", model.aic)

            print("complex model -- model summary is")
            print()
            print(model.summary())


            model_sets = []
            split_model_score_sets = []
            tree_model_score_sets = []
            tree_split_model_score_sets = []

            selected_features_sets = []
            tree_model_importance_sets = []
            tree_split_model_importance_sets = []


            for cluster_validation in [1,0]:
                for search_metric in ['vif','aic']:
                    inter_country_train_data = []
                    inter_country_train_label = []
                    fitlered_country_pair_num = 0
                    fitlered_sampling_pair_num = 0

                    country_pair_syn_dict_dict_list = defaultdict(lambda: defaultdict(list))
                    for cur_alpha3_1 in country_pair_syn_dict_dict_dict:
                        for cur_alpha3_2 in country_pair_syn_dict_dict_dict[cur_alpha3_1]:
                            cur_country_pair = copy.deepcopy(country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2])

                            cur_country_pair_simple_lang_vec = cur_country_pair["simple_lang_vec"]
                            cur_country_pair_lang_vec = cur_country_pair["lang_vec"]
                            cur_country_pair_sim = cur_country_pair["sim"]
                            cur_country_pair_same_lang = cur_country_pair["same_lang"]
                            cur_country_pair_same_spec_lang = cur_country_pair["same_spec_lang"]

                            cur_country_pair.pop('simple_lang_vec')
                            cur_country_pair.pop('lang_vec')
                            cur_country_pair.pop('sim')
                            cur_country_pair.pop('same_lang')
                            cur_country_pair.pop('same_spec_lang')


                            if cur_country_pair['pair_count'] < min_pair_count:
                                continue
                            if np.isnan(cur_country_pair['border']):
                                continue
                            if np.isnan(cur_country_pair['diplomatic']):
                                continue
                            if np.isnan(cur_country_pair['investment']):
                                continue
                            if np.isnan(cur_country_pair['trade']):
                                continue
                            if np.isnan(cur_country_pair['immgration']):
                                continue

                            for attr in cur_country_pair.values():
                                if isinstance(attr, int) or isinstance(attr, float):
                                    country_pair_syn_dict_dict_list[cur_alpha3_1][cur_alpha3_2].append(attr)
                                # if isinstance(attr, list) and len(attr) == 1:
                                if isinstance(attr, list):
                                    country_pair_syn_dict_dict_list[cur_alpha3_1][cur_alpha3_2] += attr

                            try:
                                if subsampling:
                                    for samp in range(subsampling_num):
                                        cur_samp = random.randint(0, len(cur_country_pair_sim)-1)
                                        cur_train_data_row = country_pair_syn_dict_dict_list[cur_alpha3_1][cur_alpha3_2] + [cur_country_pair_same_lang[cur_samp]] + cur_country_pair_same_spec_lang[cur_samp]

                                        try:
                                            inter_country_train_data.append(cur_train_data_row)

                                            if cluster_validation:
                                                inter_country_train_label.append(1-jensenshannon_dict[cur_alpha3_1][cur_alpha3_2])
                                            else:
                                                inter_country_train_label.append((cur_country_pair["avg_sim"]-1)/3)
                                                # inter_country_train_label.append(cur_country_pair_sim[cur_samp])
                                                # inter_country_train_label.append(4 - cur_country_pair["sim"][cur_samp])

                                            # if cluster_validation:
                                            #     inter_country_train_label.append(jensenshannon_dict[cur_alpha3_1][cur_alpha3_2])
                                            # else:
                                            #     # Similarity
                                            #     inter_country_train_label.append(cur_country_pair['avg_sim'])
                                            #     # # similarity
                                            #     # inter_country_train_label.append(4 - cur_country_pair['avg_sim'])
                                            # inter_country_train_data.append(country_pair_syn_dict_dict_list[cur_alpha3_1][cur_alpha3_2])
                                        except:
                                            fitlered_sampling_pair_num += 1
                                            traceback.print_exc()
                                else:
                                    cur_train_data_row = country_pair_syn_dict_dict_list[cur_alpha3_1][cur_alpha3_2] + cur_country_pair_simple_lang_vec + cur_country_pair_lang_vec
                                    inter_country_train_data.append(cur_train_data_row)

                                    if cluster_validation:
                                        inter_country_train_label.append(1-jensenshannon_dict[cur_alpha3_1][cur_alpha3_2])
                                    else:
                                        inter_country_train_label.append((cur_country_pair["avg_sim"]-1)/3)
                            except:
                                fitlered_country_pair_num += 1
                    print("fitlered_country_pair_num: ", fitlered_country_pair_num)
                    print("fitlered_sampling_pair_num: ", fitlered_sampling_pair_num)

                    # normalization
                    minmax = MinMaxScaler()
                    inter_country_train_data = minmax.fit_transform(inter_country_train_data)

                    print("country pairs total number:", len(inter_country_train_label))

                    inter_country_train_data_df = pd.DataFrame(inter_country_train_data)
                    inter_country_train_data_df.columns = ['pair_count',"avg_sim","Neighbor", "Geography distance","Continent sim",\
                                                        'Same political group','Both in NATO', 'Both in EU', 'Both in BRICS',\
                                                       'Across NATO-EU groups', 'Across NATO-BRICS groups', "Across EU-BRICS groups",\
                                                       "Same political system type", "Both republic", "Both federalism", \
                                                       "Neither republic nor federalism", "Republic and federalism", 'Republic and other', 'Federalism and other', \
                                                       "Same GDP category", "GDP(low-low)", "GDP(high-high)", "GDP(low-high)", \
                                                       "Same Gini Index category", "Gini Index(low-low)", "Gini Index(high-high)", "Gini Index(low-high)", \
                                                       "Same Democracy index category", "Democracy index(low-low)", "Democracy index(high-high)", "Democracy index(low-high)",\
                                                       "Diplomatic Relation", "Investment", "Trade", "Immigration", \
                                                       "Same language", "Speaking English", "Speaking German", "Speaking Spanish", "Speaking Polish", "Speaking French", \
                                                       "Speaking Chinese", "Speaking Arabic", "Speaking Turkish", "Speaking Italian", "Speaking Russian"]


                    # 0 is the pair number, 1 is the average similarity
                    # y = inter_country_train_data_df[1]
                    y = inter_country_train_label
                    X = inter_country_train_data_df.drop(['pair_count',"avg_sim"], axis=1)

                    X_feature_names = X.columns.values.tolist()
                    valid_X_feature_names = []
                    # filter the columns with only 1 kind of value, e.g. all 0 or all 1
                    for cur_feature in X_feature_names:
                        multivalue_sign = 0

                        cur_feature_values = X[cur_feature]
                        cmp_value = cur_feature_values[0]
                        for cur_feature_value in cur_feature_values:
                            if cur_feature_value != cmp_value:
                                multivalue_sign = 1
                                break
                        if multivalue_sign == 1:
                            valid_X_feature_names.append(cur_feature)

                    X = X[valid_X_feature_names]

                    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

                    # Create an empty dictionary that will be used to store our results
                    function_dict = {'predictor': [], 'r-squared': []}
                    # Iterate through every column in X
                    cols = list(X.columns)
                    for col in cols:
                        # Create a dataframe called selected_X with only the 1 column
                        selected_X = X[[col]]
                        # Fit a model for our target and our selected column
                        model = sm.OLS(y, sm.add_constant(selected_X)).fit()
                        # Predict what our target would be for our model
                        y_preds = model.predict(sm.add_constant(selected_X))
                        # Add the column name to our dictionary
                        function_dict['predictor'].append(col)
                        # Calculate the r-squared value between the target and predicted target
                        r2 = np.corrcoef(y, y_preds)[0, 1] ** 2
                        # Add the r-squared value to our dictionary
                        function_dict['r-squared'].append(r2)

                    # Once it's iterated through every column, turn our dictionary into a DataFrame and sort it
                    function_df = pd.DataFrame(function_dict).sort_values(by=['r-squared'], ascending=False)
                    # Display only the top 5 predictors

                    if search_metric == "vif":
                        selected_features = [function_df['predictor'].iat[0]]
                        features_to_ignore = []

                        # Since our function's ignore_features list is already empty, we don't need to
                        # include our features_to_ignore list.
                        while len(selected_features) + len(features_to_ignore) < len(cols):
                            next_feature = next_possible_feature(X_npf=X, y_npf=y, current_features=selected_features,
                                                                     ignore_features=features_to_ignore)[0]
                            # check vif score
                            vif_factor = 5
                            temp_selected_features = selected_features + [next_feature]
                            temp_X = X[temp_selected_features]
                            temp_vif = pd.DataFrame()
                            temp_vif["features"] = temp_X.columns
                            temp_vif["VIF"] = [variance_inflation_factor(temp_X.values, i) for i in range(len(temp_X.columns))]
                            cur_vif = temp_vif["VIF"].iat[-1]
                            if cur_vif <= vif_factor:
                                selected_features = temp_selected_features
                            else:
                                features_to_ignore.append(next_feature)
                    elif search_metric == "aic":
                        selected_features = [function_df['predictor'].iat[0]]
                        rest_features = []
                        for cur_feature in valid_X_feature_names:
                            if cur_feature not in selected_features:
                                rest_features.append(cur_feature)

                        best_aic = 10000000
                        search_max_time = 10000
                        search_time = 0
                        while len(selected_features) < len(cols) or search_time >= search_max_time:
                            # if there is no change in this turn then meaning no feature is selected.
                            # Should also stop search in this case
                            change_sign = 0
                            temp_feature_sets = [selected_features+[temp_feature] for temp_feature in rest_features]
                            for temp_feature_set in temp_feature_sets:
                                temp_X = X[temp_feature_set]
                                temp_model = sm.OLS(y, sm.add_constant(temp_X)).fit()
                                if temp_model.aic < best_aic:
                                    best_aic = temp_model.aic
                                    selected_features = temp_feature_set
                                    change_sign = 1
                            if change_sign == 0:
                                break
                            search_time += 1


                    model = sm.OLS(y, sm.add_constant(X[selected_features])).fit()
                    model_sets.append(model)

                    split_model = sm.OLS(y_train, sm.add_constant(X_train[selected_features])).fit()


                    cur_X_input = X_test[selected_features].copy()
                    cur_X_input.insert(0,'const',1)
                    split_model_score_sets.append(r2_score(y_test, split_model.predict(cur_X_input)))

                    selected_features_sets.append(selected_features)

                    tree_model = GradientBoostingRegressor(random_state=0).fit(X[selected_features], y)

                    tree_model_score_sets.append(tree_model.score(X[selected_features], y))
                    tree_model_importance_sets.append(tree_model.feature_importances_)

                    tree_split_model = GradientBoostingRegressor(random_state=0).fit(X_train[selected_features], y_train)
                    tree_split_model_score_sets.append(tree_split_model.score(X_test[selected_features], y_test))
                    tree_split_model_importance_sets.append(tree_split_model.feature_importances_)


                    M = len(selected_features)
                    xs = [np.ones(M), np.zeros(M)]
                    present_df = pd.DataFrame()
                    for idx, x in enumerate(xs):
                        index = pd.MultiIndex.from_product([[f"Example {idx}"], ["x", "shap_values"]])
                        present_df = pd.concat(
                            [
                                present_df,
                                pd.DataFrame(
                                    [x, shap.TreeExplainer(tree_split_model).shap_values(x)],
                                    index=index,
                                    columns=selected_features,
                                ),
                            ]
                        )

                    print(present_df.to_string())

                    # print(model.summary())
                    # print(model.summary().as_latex())


            '''can do the same or similar with Stargazer package!'''
            '''pealse refer to: https://pypi.org/project/stargazer/'''
            model_sets_res = summary_col(model_sets,stars=True)
            print(model_sets_res)
            print(model_sets_res.as_latex())

            print()
            print("LR split R^2")
            print()
            for split_model_score in split_model_score_sets:
                print("LR split R^2:  ", split_model_score)
                print()


            print()
            print("tree method result")
            print()

            for tree_model_score in tree_model_score_sets:
                print("R^2:  ", tree_model_score)
                print()

            for i in range(len(tree_model_importance_sets)):
                for j in range(1, len(tree_model_importance_sets[i])):
                    print(selected_features_sets[i][j], ":  ", tree_model_importance_sets[i][j])

            for tree_split_model_score in tree_split_model_score_sets:
                print("split R^2:  ", tree_split_model_score)
                print()

            for i in range(len(tree_split_model_importance_sets)):
                for j in range(1, len(tree_split_model_importance_sets[i])):
                    print(selected_features_sets[i][j], ":  ", tree_split_model_importance_sets[i][j])


        print()
        print()
        print(check_lang_country)
        print()
        print("finish at ", datetime.now())

    # regression model for intra country pairs
    # args.cluster_sim can't be used at the same time as args.opt == "rm" or args.opt == "rm_intra"
    if args.option == "rm_intra":
        filtered_out = 1
        # filtered_out = 0

        time_bin = 180
        # time_bin = 60
        # time_bin = 12

        cluster_validation = 1
        # cluster_validation = 0

        search_metric = "vif"
        # search_metric = "aic"

        subsampling = 1
        # subsampling = 0
        subsampling_num = 1000

        min_pair_count = 20
        top_k_param = 10

        country_cls_df = pd.read_csv("network_data/country_cluster/country_cluster.csv")
        country_cls_dict = country_cls_df.to_dict()
        country_cls_list = defaultdict(list)
        for i in range(country_cls_df.shape[0]):
            for cls in country_cls_dict:
                if cls == "Unnamed: 0":
                    alpha3 = country_cls_dict[cls][i]
                else:
                    country_cls_list[alpha3].append(country_cls_dict[cls][i])
        for alpha3 in country_cls_list:
            print(sum(country_cls_list[alpha3]))

        for cur_bin in range(1, int(180/time_bin)+1):
            print("***********************")
            print("***********************")
            print("cur bin is ", cur_bin)
            print("***********************")
            print("***********************")

            rm_pairs = pairs.dropna(subset=["longitude1", "longitude2", "latitude1", "latitude2"])
            rm_pairs = rm_pairs.loc[rm_pairs["main_country1"] == rm_pairs["main_country2"]]
            rm_pairs["time_bin"] = rm_pairs.apply(lambda x: round((x['date1'] + x['date2']) / 2 / time_bin), axis=1)
            if time_bin != 180:
                rm_pairs = rm_pairs[rm_pairs["time_bin"] == cur_bin]
            # rm_pairs = rm_pairs[rm_pairs["time_bin"] == cur_bin]




            # filtering the intra-country pairs whose country official language in not covered in the 10 languages of our annotation
            country_official_languages = pd.read_csv("country_info/country_official_language.csv")
            country_official_languages["Country"] = country_official_languages.apply(lambda x: x['Country'].replace("\xa0", ""), axis=1)
            country_official_languages["Official language"] = country_official_languages.apply(lambda x: x['Official language'].replace("\xa0", "").replace("\u2028", "\n").split("\n"), axis=1)
            country_official_languages["Official language family"] = country_official_languages.apply(lambda x: [LANG_FAMILY[LANG_FULL_NAME_MAP[lang]] for lang in x['Official language'] if lang in LANG_FULL_NAME_MAP], axis=1)

            country_official_languages = pd.merge(country_geography_list, country_official_languages, how='left',on='Country')

            '''load country media number '''
            rm_pairs["same_media"] = rm_pairs.apply(lambda x: 1 if x['media_id1'] == x['media_id2'] else 0, axis=1)
            rm_pairs["same_lang"] = rm_pairs.apply(lambda x: 1 if x['language1'] == x['a2_language'] else 0, axis=1)

            country_media_set = defaultdict(set)
            for pair in rm_pairs.itertuples():
                cur_alpha3_1 = pair[17]
                cur_alpha3_2 = pair[18]
                cur_media_id1 = pair[6]
                cur_media_id2 = pair[12]
                if cur_media_id1 not in country_media_set[cur_alpha3_1]:
                    country_media_set[cur_alpha3_1].add(cur_media_id1)
                if cur_media_id2 not in country_media_set[cur_alpha3_2]:
                    country_media_set[cur_alpha3_2].add(cur_media_id2)
            country_media_num = defaultdict(list)
            for cur_alpha3 in country_media_set:
                country_media_num["Alpha-3 code"].append(cur_alpha3)
                country_media_num["media_num"].append(len(country_media_set[cur_alpha3]))
            country_media_num = pd.DataFrame(country_media_num)
            country_media_num = pd.merge(country_official_languages, country_media_num, how='left',on='Alpha-3 code')

            '''load country neighbor/border info'''
            country_neighbors = pd.read_csv("country_info/country_neighbors.csv")
            country_neighbors["name"] = country_neighbors.apply(lambda x: x["name"].split(",")[0], axis=1)
            country_neighbors["borders"] = country_neighbors.apply(lambda x: x["borders"].split(",") if isinstance(x["borders"],str) else [], axis=1)
            country_neighbors = country_neighbors.drop(["status", "currencies", "capital", "region", "subregion", "languages", "latlng", "area", "demonyms"],axis=1)
            country_neighbors.rename(columns={'name': 'Country'}, inplace=True)
            country_neighbors = pd.merge(country_media_num, country_neighbors, how='left',on='Country')

            '''load continent info'''
            country_continent = pd.read_csv("country_info/country_continent.csv")
            country_continent = pd.merge(country_neighbors, country_continent, how='left', on='Country')

            '''load democracy index info'''
            country_democracy_index_list = pd.read_csv("bias_dataset/2019_democracy_index/2019_democracy_index.csv")
            country_democracy_index_list = country_democracy_index_list[country_democracy_index_list["year"] == 2019]
            country_democracy_index_list.rename(columns={'country': 'Country'}, inplace=True)
            country_democracy_index_list = pd.merge(country_continent, country_democracy_index_list, how='left', on='Country')

            '''load press freedom index info'''
            country_press_freedom_index_list = pd.read_csv("country_info/country_press_freedom_index.csv")
            country_press_freedom_index_list = pd.merge(country_democracy_index_list, country_press_freedom_index_list, how='left', on='Country')

            '''load unitary state info'''
            country_unitary_state_list = pd.read_csv("country_info/country_unitary_state.csv")
            country_unitary_state_list = pd.merge(country_press_freedom_index_list, country_unitary_state_list, how='left', on='Country')

            '''load gdp info'''
            country_gdp_list = pd.read_csv("country_info/country_gdp.csv")
            country_gdp_list = country_gdp_list[["Country Name", "2020"]]
            country_gdp_list.rename(columns={"Country Name": 'Country'}, inplace=True)
            country_gdp_list.rename(columns={"2020": '2020_gdp'}, inplace=True)

            country_gdp_list = pd.merge(country_unitary_state_list, country_gdp_list, how='left', on='Country')

            country_gdp_per_person_list = pd.read_csv("country_info/country_gdp_per_person.csv")
            country_gdp_per_person_list = country_gdp_per_person_list[["Country Name", "2020"]]
            country_gdp_per_person_list.rename(columns={"Country Name": 'Country'}, inplace=True)
            country_gdp_per_person_list.rename(columns={"2020": '2020_gdp_per_person'}, inplace=True)

            country_gdp_per_person_list = pd.merge(country_gdp_list, country_gdp_per_person_list, how='left', on='Country')

            '''load gini index info'''
            country_gini_index_list = pd.read_csv("country_info/country_gini_index.csv")
            country_gini_index_list = pd.merge(country_gdp_per_person_list, country_gini_index_list, how='left', on='Alpha-3 code')

            '''load it user rate info'''
            country_it_user_list = pd.read_csv("country_info/country_it_user.csv")
            country_it_user_list = country_it_user_list[["Country or area", "Pct"]]
            country_it_user_list.rename(columns={"Country or area": 'Country'}, inplace=True)
            country_it_user_list.rename(columns={"Pct": 'it_user_rate'}, inplace=True)
            country_it_user_list['Country'] = country_it_user_list.apply(lambda x: x['Country'].replace("\xa0",""), axis=1)
            country_it_user_list['it_user_rate'] = country_it_user_list.apply(lambda x: float(x['it_user_rate'].replace("%","")), axis=1)
            country_it_user_list = pd.merge(country_gini_index_list, country_it_user_list, how='left', on='Country')

            '''load literacy rate info'''
            country_literacy_list = pd.read_csv("country_info/country_literacy_rate.csv")
            country_literacy_list = country_literacy_list[["cca3", "latestRate"]]
            country_literacy_list.rename(columns={"cca3": 'Alpha-3 code'}, inplace=True)
            country_literacy_list.rename(columns={"latestRate": 'literacy_rate'}, inplace=True)
            country_literacy_list = pd.merge(country_it_user_list, country_literacy_list, how='left', on='Alpha-3 code')

            '''load peace index info'''
            country_peace_index_list = pd.read_csv("country_info/country_peace_index.csv")
            country_peace_index_list = country_peace_index_list[["Country", "2020 Rate"]]
            country_peace_index_list.rename(columns={"2020 Rate": 'peace_index'}, inplace=True)
            # to align the sign with other factors
            country_peace_index_list['Country'] = country_peace_index_list.apply(lambda x: x['Country'].replace("\xa0",""), axis=1)
            country_peace_index_list['peace_index'] = country_peace_index_list.apply(lambda x: -x['peace_index'], axis=1)
            country_peace_index_list = pd.merge(country_literacy_list, country_peace_index_list, how='left', on='Country')

            '''load migration info'''
            country_migration_list = pd.read_csv("country_info/migration/country_migration.csv")
            country_migration_list = country_migration_list[["Country Code", "2020"]]
            country_migration_list.rename(columns={"Country Code": 'Alpha-3 code'}, inplace=True)
            country_migration_list.rename(columns={"2020": 'net_migration'}, inplace=True)
            country_migration_list = pd.merge(country_peace_index_list, country_migration_list, how='left', on='Alpha-3 code')

            '''load religion distribution info'''
            country_religion_distribution_list = pd.read_csv("country_info/religion/country_religion_distribution.csv")
            country_religion_distribution_dict = {key: country_religion_distribution_list[key].to_list() for key in country_religion_distribution_list.keys()}
            country_religion_distribution_dict_list = defaultdict(list)
            for country_idx in range(len(country_religion_distribution_dict["Country"])):
                cur_country = country_religion_distribution_dict["Country"][country_idx]
                cur_christian = float(country_religion_distribution_dict["CHRISTIAN"][country_idx])
                cur_muslim = country_religion_distribution_dict["MUSLIM"][country_idx]
                cur_unaffil = country_religion_distribution_dict["UNAFFIL"][country_idx]
                cur_hindu = country_religion_distribution_dict["HINDU"][country_idx]
                cur_buddhist = country_religion_distribution_dict["BUDDHIST"][country_idx]
                cur_folk_religion = country_religion_distribution_dict["FOLK RELIGION"][country_idx]
                cur_other_religion = country_religion_distribution_dict["OTHER RELIGION"][country_idx]
                cur_jewish = country_religion_distribution_dict["JEWISH"][country_idx]

                country_religion_distribution_dict_list[cur_country].append(cur_christian)
                country_religion_distribution_dict_list[cur_country].append(cur_muslim)
                country_religion_distribution_dict_list[cur_country].append(cur_unaffil)
                country_religion_distribution_dict_list[cur_country].append(cur_hindu)
                country_religion_distribution_dict_list[cur_country].append(cur_buddhist)
                country_religion_distribution_dict_list[cur_country].append(cur_folk_religion)
                country_religion_distribution_dict_list[cur_country].append(cur_other_religion)
                country_religion_distribution_dict_list[cur_country].append(cur_jewish)

            country_religion_distribution_entropy_dict = {}
            for cur_country in country_religion_distribution_dict_list:
                country_religion_distribution_entropy_dict[cur_country] = (1 - entropy(country_religion_distribution_dict_list[cur_country]) / np.log(len(country_religion_distribution_dict_list[cur_country])))
            country_religion_distribution_list["religion_entropy"] = country_religion_distribution_list.apply(lambda x: country_religion_distribution_entropy_dict[x["Country"]], axis=1)
            country_religion_distribution_list = country_religion_distribution_list[["Country", "religion_entropy"]]
            country_religion_distribution_list = pd.merge(country_migration_list, country_religion_distribution_list, how='left', on='Country')

            '''load population and area info'''
            country_pop_area_list = pd.read_csv("country_info/country_population_area.csv")
            country_pop_area_list = country_pop_area_list[["cca3", "pop2021", "area", "density"]]
            country_pop_area_list.rename(columns={"cca3": 'Alpha-3 code'}, inplace=True)
            country_pop_area_list = pd.merge(country_religion_distribution_list, country_pop_area_list, how='left', on='Alpha-3 code')



            '''load political group info'''
            # dict for speedy query
            # one country or country pair can belong to multiple political group
            country_syn_list = country_pop_area_list

            political_group_df_dict = {}
            political_group_alpha3_dict = {}
            for group in political_group_list:
                political_group_df_dict[group] = pd.read_csv(f"country_info/political_group/{group}.csv")
                # nato csv contains some other countries
                if group == "nato":
                    political_group_df_dict["nato"] = political_group_df_dict["nato"][political_group_df_dict["nato"]["Category"] == "NATO"]

                political_group_df_dict[group].rename(columns={'Alpha-3': 'Alpha-3 code'}, inplace=True)
                political_group_df_dict[group] = political_group_df_dict[group][['Alpha-3 code']]
                political_group_df_dict[group] = political_group_df_dict[group].drop_duplicates()
                political_group_alpha3_dict[group] = {alpha3: 1 for alpha3 in political_group_df_dict[group]['Alpha-3 code'].to_list()}

                political_group_df_dict[group][group] = political_group_df_dict[group].apply(lambda x:group, axis=1)
                country_syn_list = pd.merge(country_syn_list, political_group_df_dict[group], how='left', on='Alpha-3 code')

            '''filter the countries without corresponding data'''
            country_syn_list = country_syn_list.dropna(subset=["media_num", "eiu", "Press Freedom", "2020_gdp", "gini_index", "it_user_rate", "literacy_rate", "peace_index", "net_migration", "religion_entropy", "pop2021", "area"])

            '''aggregate article pairs as per each country pair'''
            rm_pairs_dict = {key: rm_pairs[key].to_list() for key in rm_pairs.keys()}
            rm_edge_sim = defaultdict(lambda: defaultdict(list))  # edge's size (similarity)
            rm_edge_media_same = defaultdict(lambda: defaultdict(list))
            rm_edge_lang_same = defaultdict(lambda: defaultdict(list))
            for i in range(len(rm_pairs)):
                cur_country1 = rm_pairs_dict["main_country1"][i]
                cur_country2 = rm_pairs_dict["main_country2"][i]

                if cur_country1 < cur_country2:
                    first_country = cur_country1
                    second_country = cur_country2
                else:
                    first_country = cur_country2
                    second_country = cur_country1

                rm_edge_sim[first_country][second_country].append(rm_pairs_dict["similarity"][i])
                rm_edge_media_same[first_country][second_country].append(rm_pairs_dict["same_media"][i])
                rm_edge_lang_same[first_country][second_country].append(rm_pairs_dict["same_lang"][i])


            country_pair_syn_dict_dict_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
            country_syn_dict = {key: country_syn_list[key].to_list() for key in country_syn_list.keys()}
            for border_idx in range(len(country_syn_dict['borders'])):
                if not isinstance(country_syn_dict['borders'][border_idx], list):
                    country_syn_dict['borders'][border_idx] = []
            for i in range(len(country_syn_list)):
                for j in range(len(country_syn_list)):
                    if i == j:
                        cur_idx1 = i
                        cur_idx2 = i

                        cur_alpha3_1 = country_syn_dict["Alpha-3 code"][cur_idx1]
                        cur_alpha3_2 = country_syn_dict["Alpha-3 code"][cur_idx2]

                        print(cur_idx1, cur_alpha3_1)

                        if (cur_alpha3_1 not in rm_edge_sim) or (cur_alpha3_2 not in rm_edge_sim[cur_alpha3_1]):
                            continue


                        if filtered_out:
                            country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["pair_count"] = len(rm_edge_sim[cur_alpha3_1][cur_alpha3_2])
                            country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["avg_sim"] = np.mean(rm_edge_sim[cur_alpha3_1][cur_alpha3_2])
                            country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["sim"] = rm_edge_sim[cur_alpha3_1][cur_alpha3_2]

                            # country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["avg_same_media"] = np.mean(rm_edge_media_same[cur_alpha3_1][cur_alpha3_2])
                            country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["same_media"] = rm_edge_media_same[cur_alpha3_1][cur_alpha3_2]

                            # country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["avg_same_lang"] = np.mean(rm_edge_lang_same[cur_alpha3_1][cur_alpha3_2])
                            country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["same_lang"] = rm_edge_lang_same[cur_alpha3_1][cur_alpha3_2]

                        else:
                            # load filter-out article pairs data
                            geo_index_art_data = defaultdict(int)
                            for pair in indexes_stats.itertuples():
                                cur_main_country = pair[8]
                                cur_art_count = pair[4]
                                geo_index_art_data[cur_main_country] += cur_art_count

                            country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["pair_count"] = geo_index_art_data[cur_alpha3_1] * geo_index_art_data[cur_alpha3_2]
                            country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["avg_sim"] = np.mean(rm_edge_sim[cur_alpha3_1][cur_alpha3_2]) * len(rm_edge_sim[cur_alpha3_1][cur_alpha3_2]) / country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["pair_count"]
                            # missing corresponding integration for whether two articles are from the same media or not data and its average ratio here

                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["Country1"] = country_syn_dict["Country"][cur_idx1]
                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["Country2"] = country_syn_dict["Country"][cur_idx2]
                        # country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2][""]

                        # language categories
                        # About 12 categorical values = 10 languages (or less, because some languages like Polish are official in only one country) + “Not the same language but same family” + “Not the same language and different family” categories
                        try:
                            common_official_langs = list(country_syn_dict["Official language"][cur_idx1])
                        except:
                            common_official_langs = []
                        try:
                            common_official_langs_family = list(country_syn_dict["Official language family"][cur_idx1])
                        except:
                            common_official_langs_family = []


                        lang_num_vector = [len(common_official_langs), len(common_official_langs_family)]

                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["lang_num_vec"] = lang_num_vector

                        # political group categories, 4 kinds
                        common_group = []
                        for group in ['nato','eunion','brics']:
                            if country_syn_dict[group][cur_idx1] == country_syn_dict[group][cur_idx2]:
                                common_group.append(group)

                        '''disentangled political group features'''
                        # political_group_vector = [0 for i in range(4)]
                        # if ('nato' in common_group) and (len(common_group) == 1):
                        #     political_group_vector[0] = 1
                        # elif ('eunion' in common_group) and (len(common_group) == 1):
                        #     political_group_vector[1] = 1
                        # elif ('brics' in common_group) and (len(common_group) == 1):
                        #     political_group_vector[2] = 1
                        # elif ('nato' in common_group) and ('eunion' in common_group) and (len(common_group) == 2):
                        #     political_group_vector[3] = 1

                        ''' not disentangled political group features'''
                        political_group_vector = [0 for i in range(3)]
                        if ('nato' in common_group):
                            political_group_vector[0] = 1
                        elif ('eunion' in common_group):
                            political_group_vector[1] = 1
                        elif ('brics' in common_group):
                            political_group_vector[2] = 1

                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["political_group_vec"] = political_group_vector

                        # unitary type combination categories, 6 kinds
                        unitary_vector = [0 for i in range(2)]
                        cur_unitary_type = country_syn_dict["Unitary"][cur_idx1]
                        if cur_unitary_type == "unitary_republics":
                            unitary_vector[0] = 1
                        elif cur_unitary_type == "federalism":
                            unitary_vector[1] = 1
                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["unitary_vec"] = unitary_vector

                        # democracy index, press freedom index, gdp, gdp per person, population, area, density
                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["media_num"] = country_syn_dict["media_num"][cur_idx1]

                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["dem_idx"] = country_syn_dict["eiu"][cur_idx1]
                        # country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["pf_idx"] = country_syn_dict["Press Freedom"][cur_idx1]
                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["2020_gdp"] = country_syn_dict["2020_gdp"][cur_idx1]
                        # country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["2020_gdp_per_person"] = country_syn_dict["2020_gdp_per_person"][cur_idx1]
                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["gini_index"] = country_syn_dict["gini_index"][cur_idx1]
                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["it_user_rate"] = country_syn_dict["it_user_rate"][cur_idx1]
                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["literacy_rate"] = country_syn_dict["literacy_rate"][cur_idx1]
                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["peace_index"] = country_syn_dict["peace_index"][cur_idx1]
                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["net_migration"] = country_syn_dict["net_migration"][cur_idx1]
                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["religion_entropy"] = country_syn_dict["religion_entropy"][cur_idx1]

                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["2020_pop"] = country_syn_dict["pop2021"][cur_idx1]
                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["area"] =  country_syn_dict["area"][cur_idx1]
                        # country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["density"] = country_syn_dict["density"][cur_idx1]


                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]['country1'] = country_alpha3_geography[cur_alpha3_1]["country_full_name"]
                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]['country2'] = country_alpha3_geography[cur_alpha3_2]["country_full_name"]

            with open("indexes/rm_data_intra_country.json", 'w') as f:
                json.dump(country_pair_syn_dict_dict_dict, f)

            simple_intra_country_train_data = []
            intra_country_train_data = []
            intra_country_train_label = []
            pair_count_not_fit = 0
            dem_idx_not_fit = 0
            pf_idx_not_fit = 0
            gdp_not_fit = 0
            gdp_per_person_not_fit = 0
            for cur_alpha3_1 in country_pair_syn_dict_dict_dict:
                for cur_alpha3_2 in country_pair_syn_dict_dict_dict[cur_alpha3_1]:
                    cur_country_pair = copy.deepcopy(country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2])
                    if cur_country_pair['pair_count'] < min_pair_count:
                        pair_count_not_fit += 1
                        continue
                    # if np.isnan(cur_country_pair['dem_idx']):
                    #     dem_idx_not_fit += 1
                    #     continue
                    # if np.isnan(cur_country_pair['pf_idx']):
                    #     pf_idx_not_fit += 1
                    #     continue
                    # if np.isnan(cur_country_pair['2020_gdp']):
                    #     gdp_not_fit += 1
                    #     continue
                    # if np.isnan(cur_country_pair['2020_gdp_per_person']):
                    #     gdp_per_person_not_fit += 1
                    #     continue



                    if subsampling:
                        for samp in range(subsampling_num):
                            cur_samp = random.randint(0, len(cur_country_pair["sim"])-1)
                            intra_country_train_data.append(cur_country_pair['lang_num_vec'] + cur_country_pair['political_group_vec'] + cur_country_pair['unitary_vec'] + [cur_country_pair['dem_idx']] + [cur_country_pair['2020_gdp']] + [cur_country_pair['gini_index']] + [cur_country_pair['2020_pop']] + [cur_country_pair['area']] + [cur_country_pair["same_media"][cur_samp]] + [cur_country_pair["same_lang"][cur_samp]])

                            if cluster_validation:
                                intra_country_train_label.append(1-entropy(country_cls_list[cur_alpha3_1])/np.log(len(country_cls_list[cur_alpha3_1])))
                            else:
                                intra_country_train_label.append(cur_country_pair["sim"][cur_samp])
                    else:
                        for cur_idx in range(len(cur_country_pair["sim"])):
                            # intra_country_train_data.append([cur_country_pair['dist']] + [cur_country_pair['continent_sim']] + cur_country_pair['lang_vec'] + cur_country_pair['political_group_vec'] + [cur_country_pair['dem_idx_diff']] + [cur_country_pair['pf_idx_diff']])

                            intra_country_train_data.append(cur_country_pair['lang_num_vec'] + cur_country_pair['political_group_vec'] + cur_country_pair['unitary_vec'] + [cur_country_pair['dem_idx']] + [cur_country_pair['2020_gdp']] + [cur_country_pair['gini_index']] + [cur_country_pair['2020_pop']] + [cur_country_pair['area']] + [cur_country_pair["same_media"][cur_idx]] + [cur_country_pair["same_lang"][cur_idx]])


                            if cluster_validation:
                                intra_country_train_label.append(1-entropy(country_cls_list[cur_alpha3_1])/np.log(len(country_cls_list[cur_alpha3_1])))
                            else:
                                # Similarity
                                # intra_country_train_label.append(cur_country_pair['avg_sim'])
                                intra_country_train_label.append(cur_country_pair["sim"][cur_idx])

                                # # similarity
                                # intra_country_train_label.append(4 - cur_country_pair['avg_sim'])
            # normalization
            minmax = MinMaxScaler()
            intra_country_train_data = minmax.fit_transform(intra_country_train_data)
            print("country pairs total number:", len(intra_country_train_label))
            # print("pair_count_not_fit:", pair_count_not_fit)
            # print("dem_idx_not_fit:", dem_idx_not_fit)
            # print("pf_idx_not_fit:", pf_idx_not_fit)
            # print("gdp_not_fit:", gdp_not_fit)


            # model = LinearRegression()
            # model.fit(intra_country_train_data, intra_country_train_label)
            # print("parameter w={},b={}".format(model.coef_, model.intercept_))
            # model_coef_abs = [abs(coef) for coef in model.coef_]
            # top_k_param_idx = heapq.nlargest(top_k_param, range(len(model_coef_abs)), model_coef_abs.__getitem__)
            # top_k_param_weight = [model.coef_[k] for k in top_k_param_idx]



            print("-----------------------------")
            print("-----------------------------")
            print("complex model")
            print("-----------------------------")
            print("-----------------------------")

            model = sm.OLS(intra_country_train_label, sm.add_constant(intra_country_train_data)).fit()
            print("complex model -- parameter with intercept:", model.params)
            model_coef_abs = [abs(coef) for coef in model.params]
            top_k_param_idx = heapq.nlargest(top_k_param, range(len(model_coef_abs)), model_coef_abs.__getitem__)
            top_k_param_weight = [model.params[k] for k in top_k_param_idx]


            print(f"complex model -- the index of top {top_k_param} param (abs):", top_k_param_idx)
            print(f"complex model -- the weight of top {top_k_param} param:", top_k_param_weight)
            print(f"complex model -- AIC:", model.aic)

            print("complex model -- model summary is")
            print()
            print(model.summary())

            # print("model confidence interval is")
            # print()
            # print(model.conf_int(0.05))







            # "incomplete: test for selecting feature"
            model_sets = []
            split_model_score_sets = []
            tree_model_score_sets = []
            tree_split_model_score_sets = []

            selected_features_sets = []
            tree_model_importance_sets = []
            tree_split_model_importance_sets = []
            for cluster_validation in [1,0]:
                for search_metric in ['vif','aic']:
                    intra_country_train_data = []
                    intra_country_train_label = []

                    country_pair_syn_dict_dict_list = defaultdict(lambda: defaultdict(list))
                    for cur_alpha3_1 in country_pair_syn_dict_dict_dict:
                        for cur_alpha3_2 in country_pair_syn_dict_dict_dict[cur_alpha3_1]:
                            cur_country_pair = copy.deepcopy(country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2])
                            cur_country_pair_sim = cur_country_pair["sim"]
                            cur_country_pair_same_media = cur_country_pair["same_media"]
                            cur_country_pair_same_lang = cur_country_pair["same_lang"]

                            cur_country_pair.pop('sim')
                            cur_country_pair.pop('same_media')
                            cur_country_pair.pop('same_lang')
                            # if cur_country_pair['pair_count'] < min_pair_count:
                            #     continue
                            # if np.isnan(cur_country_pair['dem_idx']):
                            #     continue
                            # if np.isnan(cur_country_pair['pf_idx']):
                            #     continue
                            # if np.isnan(cur_country_pair['2020_gdp']):
                            #     continue
                            # if np.isnan(cur_country_pair['2020_gdp_per_person']):
                            #     continue
                            for attr in cur_country_pair.values():
                                if isinstance(attr, int) or isinstance(attr, float):
                                    country_pair_syn_dict_dict_list[cur_alpha3_1][cur_alpha3_2].append(attr)
                                # if isinstance(attr, list) and len(attr) == 1:
                                if isinstance(attr, list) and len(attr) > 0:
                                    country_pair_syn_dict_dict_list[cur_alpha3_1][cur_alpha3_2] += attr

                            if subsampling:
                                for samp in range(subsampling_num):
                                    try:
                                        cur_samp = random.randint(0, len(cur_country_pair_sim)-1)
                                        cur_train_data_row = country_pair_syn_dict_dict_list[cur_alpha3_1][cur_alpha3_2] + [cur_country_pair_same_media[cur_samp]] + [cur_country_pair_same_lang[cur_samp]]
                                        intra_country_train_data.append(cur_train_data_row)

                                        if cluster_validation:
                                            intra_country_train_label.append(1-entropy(country_cls_list[cur_alpha3_1])/np.log(len(country_cls_list[cur_alpha3_1])))
                                        else:
                                            # Similarity
                                            intra_country_train_label.append((cur_country_pair_sim[cur_samp]-1)/3)
                                    except:
                                        pass
                            else:
                                for cur_idx in range(len(cur_country_pair_sim)):
                                    cur_train_data_row = country_pair_syn_dict_dict_list[cur_alpha3_1][cur_alpha3_2] + [cur_country_pair_same_media[cur_idx]] + [cur_country_pair_same_lang[cur_samp]]
                                    intra_country_train_data.append(cur_train_data_row)

                                    if cluster_validation:
                                        intra_country_train_label.append(1-entropy(country_cls_list[cur_alpha3_1])/np.log(len(country_cls_list[cur_alpha3_1])))
                                    else:
                                        # Similarity
                                        intra_country_train_label.append((cur_country_pair_sim[cur_idx]-1)/3)
                                        # intra_country_train_label.append(cur_country_pair['avg_sim'])

                                        # # similarity
                                        # intra_country_train_label.append(4 - cur_country_pair['avg_sim'])
                    # normalization
                    minmax = MinMaxScaler()
                    intra_country_train_data = minmax.fit_transform(intra_country_train_data)
                    print("country pairs total number:", len(intra_country_train_label))

                    intra_country_train_data_df = pd.DataFrame(intra_country_train_data)

                    intra_country_train_data_df.columns = ['pair_count',"avg_sim","Number of language", "Number of language family",\
                                                           "Within NATO?", "Within BRICS?", "Within EU?", "Republic?", "Federalism?", "Number of media",\
                                                           "Democracy index", "GDP", "Gini index", "IT User Rate", "Literacy Rate", "Peace Index", "Net Migration", "Religion Disstribution Entropy", "Population", "Area", "Same media", "Same language?"]

                    # 0 is the pair number, 1 is the average similarity
                    y = intra_country_train_label
                    X = intra_country_train_data_df.drop(['pair_count',"avg_sim"], axis=1)

                    X_feature_names = X.columns.values.tolist()
                    valid_X_feature_names = []
                    # filter the columns with only 1 kind of value, e.g. all 0 or all 1
                    for cur_feature in X_feature_names:
                        multivalue_sign = 0

                        cur_feature_values = X[cur_feature]
                        cmp_value = cur_feature_values[0]
                        for cur_feature_value in cur_feature_values:
                            if cur_feature_value != cmp_value:
                                multivalue_sign = 1
                                break
                        if multivalue_sign == 1:
                            valid_X_feature_names.append(cur_feature)

                    X = X[valid_X_feature_names]

                    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

                    # Create an empty dictionary that will be used to store our results
                    function_dict = {'predictor': [], 'r-squared': []}
                    # Iterate through every column in X
                    cols = list(X.columns)
                    for col in cols:
                        # Create a dataframe called selected_X with only the 1 column
                        selected_X = X[[col]]
                        # Fit a model for our target and our selected column
                        model = sm.OLS(y, sm.add_constant(selected_X)).fit()
                        # Predict what our target would be for our model
                        y_preds = model.predict(sm.add_constant(selected_X))
                        # Add the column name to our dictionary
                        function_dict['predictor'].append(col)
                        # Calculate the r-squared value between the target and predicted target
                        r2 = np.corrcoef(y, y_preds)[0, 1] ** 2
                        # Add the r-squared value to our dictionary
                        function_dict['r-squared'].append(r2)

                    # Once it's iterated through every column, turn our dictionary into a DataFrame and sort it
                    function_df = pd.DataFrame(function_dict).sort_values(by=['r-squared'], ascending=False)
                    # Display only the top 5 predictors

                    if search_metric == "vif":
                        selected_features = [function_df['predictor'].iat[0]]
                        features_to_ignore = []

                        # Since our function's ignore_features list is already empty, we don't need to
                        # include our features_to_ignore list.
                        while len(selected_features) + len(features_to_ignore) < len(cols):
                            next_feature = next_possible_feature(X_npf=X, y_npf=y, current_features=selected_features,
                                                                     ignore_features=features_to_ignore)[0]
                            # check vif score
                            vif_factor = 5
                            temp_selected_features = selected_features + [next_feature]
                            temp_X = X[temp_selected_features]
                            temp_vif = pd.DataFrame()
                            temp_vif["features"] = temp_X.columns
                            temp_vif["VIF"] = [variance_inflation_factor(temp_X.values, i) for i in range(len(temp_X.columns))]
                            cur_vif = temp_vif["VIF"].iat[-1]
                            if cur_vif <= vif_factor:
                                selected_features = temp_selected_features
                            else:
                                features_to_ignore.append(next_feature)
                    elif search_metric == "aic":
                        selected_features = [function_df['predictor'].iat[0]]
                        rest_features = []
                        for cur_feature in valid_X_feature_names:
                            if cur_feature not in selected_features:
                                rest_features.append(cur_feature)

                        best_aic = 10000000
                        search_max_time = 10000
                        search_time = 0
                        while len(selected_features) < len(cols) or search_time >= search_max_time:
                            # if there is no change in this turn then meaning no feature is selected.
                            # Should also stop search in this case
                            change_sign = 0
                            temp_feature_sets = [selected_features+[temp_feature] for temp_feature in rest_features]
                            for temp_feature_set in temp_feature_sets:
                                temp_X = X[temp_feature_set]
                                temp_model = sm.OLS(y, sm.add_constant(temp_X)).fit()
                                if temp_model.aic < best_aic:
                                    best_aic = temp_model.aic
                                    selected_features = temp_feature_set
                                    change_sign = 1
                            if change_sign == 0:
                                break
                            search_time += 1


                    model = sm.OLS(y, sm.add_constant(X[selected_features])).fit()
                    model_sets.append(model)

                    split_model = sm.OLS(y_train, sm.add_constant(X_train[selected_features])).fit()

                    cur_X_input = X_test[selected_features].copy()
                    cur_X_input.insert(0, 'const', 1)
                    split_model_score_sets.append(r2_score(y_test, split_model.predict(cur_X_input)))

                    selected_features_sets.append(selected_features)

                    tree_model = GradientBoostingRegressor(random_state=0).fit(X[selected_features], y)
                    tree_model_score_sets.append(tree_model.score(X[selected_features], y))
                    tree_model_importance_sets.append(tree_model.feature_importances_)

                    tree_split_model = GradientBoostingRegressor(random_state=0).fit(X_train[selected_features],
                                                                                     y_train)
                    tree_split_model_score_sets.append(tree_split_model.score(X_test[selected_features], y_test))
                    tree_split_model_importance_sets.append(tree_split_model.feature_importances_)

                    M = len(selected_features)
                    xs = [np.ones(M), np.zeros(M)]
                    present_df = pd.DataFrame()
                    for idx, x in enumerate(xs):
                        index = pd.MultiIndex.from_product([[f"Example {idx}"], ["x", "shap_values"]])
                        present_df = pd.concat(
                            [
                                present_df,
                                pd.DataFrame(
                                    [x, shap.TreeExplainer(tree_split_model).shap_values(x)],
                                    index=index,
                                    columns=selected_features,
                                ),
                            ]
                        )

                    print(present_df.to_string())

                    # print(model.summary())
                    # print(model.summary().as_latex())

            model_sets_res = summary_col(model_sets, stars=True)
            print(model_sets_res)
            print(model_sets_res.as_latex())

            print()
            print("LR split R^2")
            print()
            for split_model_score in split_model_score_sets:
                print("LR split R^2:  ", split_model_score)
                print()

            print()
            print("tree method result")
            print()

            for tree_model_score in tree_model_score_sets:
                print("R^2:  ", tree_model_score)
                print()

            for i in range(len(tree_model_importance_sets)):
                for j in range(1, len(tree_model_importance_sets[i])):
                    print(selected_features_sets[i][j], ":  ", tree_model_importance_sets[i][j])

            for tree_split_model_score in tree_split_model_score_sets:
                print("split R^2:  ", tree_split_model_score)
                print()

            for i in range(len(tree_split_model_importance_sets)):
                for j in range(1, len(tree_split_model_importance_sets[i])):
                    print(selected_features_sets[i][j], ":  ", tree_split_model_importance_sets[i][j])

        print()
        print("finish at ", datetime.now())

    if args.option == "oslom_rm":
        filtered_out = 1
        # filtered_out = 0

        time_bin = 180
        # time_bin = 60
        # time_bin = 12

        min_pair_count = 100
        top_k_param = 10

        min_cls_size = 100

        # clustering_info_route = "network_data/input/backup/oslom.tsv_oslo_files/partitions_level_0"
        clustering_info_route = "network_data/input/oslom.tsv_oslo_files/partitions_level_0"

        for cur_bin in range(1, int(180/time_bin)+1):
            print("***********************")
            print("***********************")
            print("cur bin is ", cur_bin)
            print("***********************")
            print("***********************")

            '''this section is only used for first time to get the link data '''
            # rm_pairs = pairs.dropna(subset=["longitude1", "longitude2", "latitude1", "latitude2"])
            # rm_pairs["time_bin"] = rm_pairs.apply(lambda x: round((x['date1'] + x['date2']) / 2 / time_bin), axis=1)
            # if time_bin != 180:
            #     rm_pairs = rm_pairs[rm_pairs["time_bin"] == cur_bin]
            #
            # rm_pairs_network_input = rm_pairs[["stories_id1","stories_id2","similarity"]]
            # rm_pairs_network_input.to_csv("network_data/input/oslom.csv")
            #
            # pd_all = pd.read_csv("network_data/input/oslom.csv")
            # pd_all.to_csv('network_data/input/oslom.tsv', sep='\t', header=None, index=False, columns=["stories_id1","stories_id2","similarity"], mode="w")

            '''load cluster data by oslom'''
            # art_dict = defaultdict(dict)

            # for pair in pairs.itertuples():
            #     cur_date1 = pair[3]
            #     cur_id1 = pair[4]
            #     cur_lang1 = pair[5]
            #     cur_media_id1 = pair[6]
            #     cur_media_name1 = pair[7]
            #     cur_media_url1 = pair[8]
            #
            #     cur_date2 = pair[9]
            #     cur_id2 = pair[10]
            #     cur_lang2 = pair[11]
            #     cur_media_id2 = pair[12]
            #     cur_media_name2 = pair[13]
            #     cur_media_url2 = pair[14]
            #
            #     cur_lang_family1 = pair[15]
            #     cur_lang_family2 = pair[16]
            #     cur_pub_country1 = pair[17]
            #     cur_pub_country2 = pair[18]
            #     cur_about_country1 = pair[19]
            #     cur_about_country2 = pair[20]
            #     cur_bias1 = pair[21]
            #     cur_bias2 = pair[22]
            #     cur_bias_country1 = pair[23]
            #     cur_bias_country2 = pair[24]
            #     cur_main_country1 = pair[25]
            #     cur_main_country2 = pair[26]
            #
            #     cur_country_full_name1 = pair[27]
            #     cur_latitude1 = pair[28]
            #     cur_longitude1 = pair[29]
            #     cur_democracy_index1 = pair[30]
            #
            #     cur_country_full_name2 = pair[31]
            #     cur_latitude2 = pair[32]
            #     cur_longitude2 = pair[33]
            #     cur_democracy_index2 = pair[35]
            #
            #     cur_country_gdp1 = pair[36]
            #     cur_country_gdp2 = pair[37]
            #
            #     cur_country_gdp1_per_person = pair[38]
            #     cur_country_gdp2_per_person = pair[39]
            #
            #
            #     art_dict[cur_id1]["date"] = cur_date1
            #     art_dict[cur_id1]["sotries_id"] = cur_id1
            #     art_dict[cur_id1]["language"] = cur_lang1
            #     art_dict[cur_id1]["media_id"] = cur_media_id1
            #     art_dict[cur_id1]["media_name"] = cur_media_name1
            #     art_dict[cur_id1]["media_url"] = cur_media_url1
            #
            #     art_dict[cur_id2]["date"] = cur_date2
            #     art_dict[cur_id2]["sotries_id"] = cur_id2
            #     art_dict[cur_id2]["language"] = cur_lang2
            #     art_dict[cur_id2]["media_id"] = cur_media_id2
            #     art_dict[cur_id2]["media_name"] = cur_media_name2
            #     art_dict[cur_id2]["media_url"] = cur_media_url2
            #
            #     art_dict[cur_id1]["lang_family"] = cur_lang_family1
            #     art_dict[cur_id2]["lang_family"] = cur_lang_family2
            #
            #     art_dict[cur_id1]["pub_country"] = cur_pub_country1
            #     art_dict[cur_id2]["pub_country"] = cur_pub_country2
            #     art_dict[cur_id1]["about_country"] = cur_about_country1
            #     art_dict[cur_id2]["about_country"] = cur_about_country2
            #
            #     art_dict[cur_id1]["bias"] = cur_bias1
            #     art_dict[cur_id2]["bias"] = cur_bias2
            #
            #     art_dict[cur_id1]["bias_country"] = cur_bias_country1
            #     art_dict[cur_id2]["bias_country"] = cur_bias_country2
            #
            #     art_dict[cur_id1]["main_country"] = cur_main_country1
            #     art_dict[cur_id2]["main_country"] = cur_main_country2

            '''load art as per country'''
            art_dict = defaultdict(dict)

            for pair in pairs.itertuples():
                cur_id1 = pair[4]
                cur_id2 = pair[10]
                cur_main_country1 = pair[25]
                cur_main_country2 = pair[26]

                art_dict[cur_id1]["main_country"] = cur_main_country1
                art_dict[cur_id2]["main_country"] = cur_main_country2

            country_cls_dict = defaultdict(lambda: defaultdict(int))
            country_cls_list = defaultdict(list)
            total_cls_num = 0
            total_pair_num = 0
            with open(clustering_info_route, "r") as fh:
                line = fh.readline()
                cur_cls_num = 0
                while line:
                    line = fh.readline()
                    if len(line) <= 0:
                        continue
                    if line[0] == '#':
                        cur_cls_num = int(line.split(" ")[1])
                        continue

                    cur_cls = line.split(" ")
                    if len(cur_cls) < min_cls_size:
                        continue

                    total_cls_num += 1
                    total_pair_num += len(cur_cls) * (len(cur_cls) - 1)

                    for art_id in cur_cls:
                        if art_id == '\n':
                            continue
                        art_id = int(art_id)
                        try:
                            country_cls_dict[art_dict[art_id]["main_country"]][total_cls_num] += 1
                        except:
                            pass
                for alpha3 in country_cls_dict:
                    country_cls_list[alpha3] = [0 for k in range(total_cls_num)]
                    for cls in country_cls_dict[alpha3]:
                        country_cls_list[alpha3][cls-1] = country_cls_dict[alpha3][cls]
                    total_cls_count = sum(country_cls_list[alpha3])
                    country_cls_list[alpha3] = [v/total_cls_count for v in country_cls_list[alpha3]]

                print("filtered_in cluster number: ", total_cls_num)
                print("filtered_in pair number: ", total_pair_num)

                country_cls_df = pd.DataFrame(country_cls_list).T
                country_cls_df.to_csv("network_data/country_cluster/country_cluster.csv")

    #
    if args.option == "oslom_stats":
        min_cls_size = 10
        # min_cls_size = 100

        clustering_info_route = "network_data/input/oslom.tsv_oslo_files/partitions_level_0"

        art_date_dict = {}
        art_lang_dict = {}


        for pair in pairs.itertuples():
            cur_id1 = pair[4]
            cur_id2 = pair[10]

            cur_date1 = pair[3]
            cur_date2 = pair[9]
            cur_lang1 = pair[5]
            cur_lang2 = pair[11]
            cur_url1 = pair[8]
            cur_url2 = pair[14]

            art_date_dict[cur_id1] = cur_date1
            art_date_dict[cur_id2] = cur_date2
            art_lang_dict[cur_id1] = cur_lang1
            art_lang_dict[cur_id2] = cur_lang2



        url_data_path = "network_pairs/candidates/*/*"
        art_url_dict = {}
        art_title_dict = {}
        art_file_dict = {}
        art_lineno_dict = {}
        for lang in lang_list:
            read_story_wiki_data(f"indexes/{lang}-wiki-v5.index",art_url_dict, art_title_dict, art_file_dict, art_lineno_dict)
        print("art_url_dict: ", len(art_url_dict))
        print("art_title_dict: ", len(art_title_dict))


        art_cls_dict = defaultdict(set)
        cls_art_dict = defaultdict(set)
        cls_date_lists = defaultdict(list)
        cls_lang_sets = defaultdict(set)
        cls_namelist = []
        cls_sizes = {}
        with open(clustering_info_route, "r") as fh:
            line = fh.readline()
            cur_cls_num = 0
            while line:
                line = fh.readline()
                if len(line) <= 0:
                    continue
                if line[0] == '#':
                    cur_cls_num = int(line.split(" ")[1])
                    continue

                cur_cls = line.split(" ")
                if len(cur_cls) < min_cls_size:
                    continue
                cls_namelist.append(cur_cls_num)
                cls_sizes[cur_cls_num] = len(cur_cls)


                for art_id in cur_cls:
                    if art_id == '\n':
                        continue

                    art_id = int(art_id)
                    art_cls_dict[art_id].add(cur_cls_num)
                    cls_art_dict[cur_cls_num].add(art_id)

                    try:
                        cls_date_lists[cur_cls_num].append(art_date_dict[art_id])
                    except:
                        pass

                    try:
                        cls_lang_sets[cur_cls_num].add(art_lang_dict[art_id])
                    except:
                        pass




        cls_size_class = {10: 0, 50: 0, 100: 0, 200: 0, 300: 0, 500: 0, 1000: 0}
        cls_date_class = {1:0, 2:0, 3:0, 7:0, 15:0, 30:0, 90:0, 180:0}
        cls_lang_class = {1:0, 2:0, 3:0, 5:0, 10:0}
        for cur_cls_num in cls_namelist:
            try:
                cur_cls_size = cls_sizes[cur_cls_num]
                for size_class in cls_size_class:
                    if cur_cls_size <= size_class:
                        cls_size_class[size_class] += 1
                        break
            except:
                pass

            try:
                cur_cls_date_span = max(cls_date_lists[cur_cls_num]) - min(cls_date_lists[cur_cls_num])
                for date_class in cls_date_class:
                    if cur_cls_date_span <= date_class:
                        cls_date_class[date_class] += 1
                        break
            except:
                pass

            try:
                cur_lang_num = len(cls_lang_sets[cur_cls_num])
                for lang_class in cls_lang_class:
                    if cur_lang_num <= lang_class:
                        cls_lang_class[lang_class] += 1
                        break
            except:
                pass



        # figure
        fig = plt.figure(figsize=(18, 14.4))

        value = list(cls_size_class.values())
        name_list = ["0~10", "10~50", "50~100", "100~200", "200~300", "300~500", "500~1000"]
        # color_list = ["silver", "grey", "lightcoral", 'r', 'blue']

        plt.bar(range(len(value)), value, tick_label=name_list, width=2)
        plt.xticks(size=23)
        plt.yticks(size=23)
        plt.xlabel('Cluster size (article number)', fontsize=32)
        plt.ylabel('Cluster number', fontsize=32)

        plt.savefig("figs/cluster_validation/cls_size_class.png")

        # figure
        fig = plt.figure(figsize=(13, 10.4))

        value = list(cls_date_class.values())[:-3]
        name_list = ["0~1", "1~2", "2~3", "3~7", "7~15"]
        # color_list = ["silver", "grey", "lightcoral", 'r', 'blue']

        plt.bar(range(len(value)), value, tick_label=name_list)
        plt.xticks(size=23)
        plt.yticks(size=23)
        plt.xlabel('Cluster span (days)', fontsize=32)
        plt.ylabel('Cluster number', fontsize=32)

        plt.savefig("figs/cluster_validation/cls_date_class.png")

        # figure
        fig = plt.figure(figsize=(13, 10.4))

        value = list(cls_lang_class.values())
        name_list = ["1","2", "3", "3~5", "5~10"]
        # color_list = ["silver", "grey", "lightcoral", 'r', 'blue']

        plt.bar(range(len(value)), value, tick_label=name_list)
        plt.xticks(size=23)
        plt.yticks(size=23)
        plt.xlabel('Number of languages', fontsize=32)
        plt.ylabel('Cluster number', fontsize=32)

        plt.savefig("figs/cluster_validation/cls_lang_class.png")


        # computing modularity
        eii_dict = defaultdict(int)
        ai_dict = defaultdict(int)
        pair_sim_class = {0.2:[], 0.3:[], 0.4:[], 0.5:[], 0.6:[], 1:[]}
        node_degree_count = defaultdict(int)
        node_cluster_count = defaultdict(int)
        pair_link_dict = defaultdict(dict)
        for pair in pairs.itertuples():
            cur_id1 = pair[4]
            cur_id2 = pair[10]

            pair_link_dict[cur_id1][cur_id2] = 1
            pair_link_dict[cur_id2][cur_id1] = 1

            node_degree_count[cur_id1] += 1
            node_degree_count[cur_id2] += 1

            node_cluster_count[cur_id1] = len(art_cls_dict[cur_id1])
            node_cluster_count[cur_id2] = len(art_cls_dict[cur_id2])

            cur_pair_sim = pair[2]

            cur_intersect = set(art_cls_dict[cur_id1]).intersection(art_cls_dict[cur_id2])
            cur_union = set(art_cls_dict[cur_id1]).union(set(art_cls_dict[cur_id2]))
            for cls in cur_intersect:
                eii_dict[cls]+= 1
            for cls in cur_union:
                ai_dict[cls] += 1

            for cur_pair_sim_class in pair_sim_class:
                if cur_pair_sim<=cur_pair_sim_class:
                    if len(cur_intersect) != 0:
                        pair_sim_class[cur_pair_sim_class].append(1)
                    else:
                        pair_sim_class[cur_pair_sim_class].append(0)
        pair_len = pairs.shape[0]

        # wrong modularity, since this is for clusters that have no overlap
        # cls_modularity = 0
        # for cls in cls_namelist:
        #     cls_modularity += eii_dict[cls]/pair_len - (ai_dict[cls]/pair_len) * (ai_dict[cls]/pair_len)

        # right modularity
        cls_modularity = 0
        for cls in cls_namelist:
            for cur_id1 in cls_art_dict[cls]:
                for cur_id2 in cls_art_dict[cls]:
                    if node_cluster_count[cur_id1] != 0 and node_cluster_count[cur_id2] != 0:
                        try:
                            cls_modularity = cls_modularity + 1/(node_cluster_count[cur_id1] * node_cluster_count[cur_id2]) *(pair_link_dict[cur_id1][cur_id2] - node_degree_count[cur_id1] * node_degree_count[cur_id2]/(2*pair_len))
                        except:
                            pass
        cls_modularity /= (2*pair_len)

        print("oslom modularity is ", cls_modularity)

        # partition density
        partition_density = 0
        for cls in cls_namelist:
            partition_density += 2/pair_len * eii_dict[cls] * (eii_dict[cls] - cls_sizes[cls] + 1) / ( (cls_sizes[cls]-2) * (cls_sizes[cls]-1) )

        print("oslom partition_density is ", partition_density)

        # figure for ratio of pairs that falls into the same cluster with different similarity
        fig = plt.figure(figsize=(10, 8))

        value = [sum(value_list)/len(value_list) for value_list in list(pair_sim_class.values())]
        name_list = pair_sim_class.keys()
        # color_list = ["silver", "grey", "lightcoral", 'r', 'blue']

        plt.bar(range(len(value)), value, tick_label=name_list)
        plt.xticks(size=23)
        plt.yticks(size=23)
        plt.xlabel('Similarity', fontsize=32)
        plt.ylabel('Ratio of within-cluster pairs', fontsize=32)

        plt.savefig("figs/cluster_validation/pair_with_cls_ratio.png")


        # plot the annotation statistic as to cluster
        annotation_data = pd.read_csv('../network_inference/df_per_annotation_for_model.csv')
        annotation_data_pair_id = annotation_data["content.pair_id"].to_list()
        annotation_data_response = annotation_data["response"].to_list()
        annotation_cls_results = {"within_cluster":defaultdict(int), "between_cluster":defaultdict(int), "ratio": defaultdict(float)}
        for i in range(len(annotation_data_pair_id)):
            cur_id1, cur_id2 = annotation_data_pair_id[i].split("_")
            cur_id1 = int(cur_id1)
            cur_id2 = int(cur_id2)

            cur_response = eval(annotation_data_response[i])[4]
            if cur_response == "Other":
                continue

            if len(art_cls_dict[cur_id1].intersection(art_cls_dict[cur_id2])) > 0:
                annotation_cls_results["within_cluster"][cur_response] += 1
            elif ((len(art_cls_dict[cur_id1]) >0) and (len(art_cls_dict[cur_id2]) >0)):
                annotation_cls_results["between_cluster"][cur_response] += 1
            # intialization
            annotation_cls_results["ratio"][cur_response] = 0

        for cur_response in annotation_cls_results["ratio"]:
            print("within_cluster", annotation_cls_results["within_cluster"][cur_response])
            print("between_cluster", annotation_cls_results["between_cluster"][cur_response])
            annotation_cls_results["ratio"][cur_response] = annotation_cls_results["within_cluster"][cur_response] / (annotation_cls_results["within_cluster"][cur_response] + annotation_cls_results["between_cluster"][cur_response])

        fig = plt.figure(figsize=(13, 10.4))

        value = []
        value.append(annotation_cls_results["ratio"]["Very Dissimilar"])
        value.append(annotation_cls_results["ratio"]["Somewhat Dissimilar"])
        value.append(annotation_cls_results["ratio"]["Somewhat Similar"])
        value.append(annotation_cls_results["ratio"]["Very Similar"])

        name_list = ["Very \n Dissimilar", "Somewhat \n Dissimilar", "Somewhat \n Similar", "Very \n Similar"]
        color_list = ["paleturquoise", "aquamarine", "cyan", 'blue']

        plt.bar(range(len(value)), value, tick_label=name_list, color=color_list)
        plt.xticks(size=24)
        plt.yticks(size=24)
        plt.xlabel('Pair Kind', fontsize=24)
        plt.ylabel('Fraction of pairs within the same cluster', fontsize=24)
        # plt.title('Average similarity of different news pair kinds.')

        plt.savefig("figs/cluster_validation/label_frac_within_cluster.png")


        fig = plt.figure(figsize=(13, 10.4))

        value = [1.56, 2.18, 2.83, 3.46]
        name_list = ["Very \n Dissimilar", "Somewhat \n Dissimilar", "Somewhat \n Similar", "Very \n Similar"]
        color_list = ["paleturquoise", "aquamarine", "cyan", 'blue']

        plt.bar(range(len(value)), value, tick_label=name_list, color=color_list)
        plt.xticks(size=24)
        plt.yticks(size=24)
        plt.xlabel('Pair Kind', fontsize=24)
        plt.ylabel('Average Similarity', fontsize=24)
        # plt.title('Average similarity of different news pair kinds.')

        plt.savefig("figs/cluster_validation/aver_label_sim_cmp.png")


        ''' sample clusters for evaluation '''
        ''' 1. select the biggest 10 clusters and 20 random clusters '''
        ''' 2. randomly select 20 articles for each cluster and an article from a random cluster as the intruder '''

        # filtered small clusters to make sure sufficient samples
        max_art_samp_num = 50
        cluster_art_samp_num = 10

        filtered_cls_sizes = {}
        for i in cls_sizes:
            if cls_sizes[i] >= 20:
                filtered_cls_sizes[i] = cls_sizes[i]

        top_cls = sorted(filtered_cls_sizes, key=filtered_cls_sizes.get)[:20]
        rand_cls = random.sample(filtered_cls_sizes.keys(), 40)
        cls_samp = list(set(top_cls).union(rand_cls))
        
        print("cls_sample:", cls_samp)

        for cls in cls_samp:
            try:
                cur_art_list = random.sample(cls_art_dict[cls], max_art_samp_num)
                print(cur_art_list)
                cur_url_list = []
                cur_title_dict = {}
                for art in cur_art_list:
                    if len(cur_url_list) >= cluster_art_samp_num:
                        break

                    if art not in art_url_dict:
                        continue

                    if "latestnigeriannews" in art_url_dict[art] or "lapatriaenlinea" in art_url_dict[art]:
                        continue

                    ia_info = checkurl(art_url_dict[art])
                    try:
                        if "closest" in ia_info["ia"]["archived_snapshots"]:
                            print("art sample")
                            ia_url = ia_info["ia"]["archived_snapshots"]["closest"]["url"]
                            cur_url_list.append(ia_url)
                            cur_title_dict[ia_url] = art_title_dict[art]
                    except:
                        traceback.print_exc()



                intruder_find = 0
                intruder_cls_list = random.sample(cls_sizes.keys(), 10)
                for intruder_cls in intruder_cls_list:
                    while intruder_cls == cls:
                        intruder_cls = random.sample(cls_sizes.keys(), 1)[0]

                    intruder_art_list = random.sample(cls_art_dict[intruder_cls], 20)
                    for art in intruder_art_list:
                        if art not in art_url_dict:
                            continue

                        ia_info = checkurl(art_url_dict[art])
                        try:
                            if "closest" in ia_info["ia"]["archived_snapshots"]:
                                print("intruder")
                                intruder_ia_url = ia_info["ia"]["archived_snapshots"]["closest"]["url"]
                                intruder_title = art_title_dict[art]
                                intruder_find = 1
                                break
                        except:
                            traceback.print_exc()
                    if intruder_find == 1:
                        break



                # intruder_cls = random.sample(cls_sizes.keys(), 1)[0]
                # while intruder_cls == cls:
                #     intruder_cls = random.sample(cls_sizes.keys(), 1)[0]
                # intruder_art = random.sample(cls_art_dict[intruder_cls], 1)[0]

                cur_url_list.append(intruder_ia_url)
                print("cur_url_list: ", cur_url_list)
                cur_title_dict[intruder_ia_url] = intruder_title
                random.shuffle(cur_url_list)

                # Path to the CSV file
                cls_sample_folder_path = "figs/cluster_validation/sample/"
                if not os.path.exists(cls_sample_folder_path):
                    os.makedirs(cls_sample_folder_path)
                cls_sample_path = cls_sample_folder_path + str(cls) + '_sample.csv'

                # Write the list to a CSV file
                with open(cls_sample_path, mode='w', newline='') as csv_file:
                    csv_writer = csv.writer(csv_file)

                    # Write the list to a column
                    csv_writer.writerow([cls])  # Header
                    for url in cur_url_list:
                        csv_writer.writerow([cur_title_dict[url], url])

                print("cur_cls: ", cls)
                print("intruder:", cur_title_dict[intruder_ia_url])
                print(intruder_ia_url)
                print("")
            except:
                traceback.print_exc()

        for cls in filtered_cls_sizes:
            try:
                cur_art_list = cls_art_dict[cls]
                print(cur_art_list)
                cur_url_list = []
                cur_title_dict = {}
                for art in cur_art_list:
                    if art not in art_url_dict:
                        continue

                    if "latestnigeriannews" in art_url_dict[art] or "lapatriaenlinea" in art_url_dict[art]:
                        continue

                    print("art sample")
                    cur_url_list.append(art_url_dict[art])
                    cur_title_dict[art_url_dict[art]] = art_title_dict[art]



                # Path to the CSV file
                cls_full_art_folder_path = "figs/cluster_validation/full_art/"
                if not os.path.exists(cls_full_art_folder_path):
                    os.makedirs(cls_full_art_folder_path)
                cls_full_art_path = cls_full_art_folder_path + str(cls) + '_full_art.csv'

                # Write the list to a CSV file
                with open(cls_full_art_path, mode='w', newline='') as csv_file:
                    csv_writer = csv.writer(csv_file)

                    # Write the list to a column
                    csv_writer.writerow([cls])  # Header
                    for url in cur_url_list:
                        csv_writer.writerow([cur_title_dict[url], url])

                print("cur_cls: ", cls)
                print("")
            except:
                traceback.print_exc()


        # # get full information of some clusters
        # top_cls = sorted(filtered_cls_sizes, key=filtered_cls_sizes.get)[:10]
        # rand_cls = random.sample(filtered_cls_sizes.keys(), 20)
        # cls_samp = list(set(top_cls).union(rand_cls))
        #
        # print("cls_sample:", cls_samp)
        #
        # for cls in cls_samp:
        #     try:
        #         cur_art_list = cls_art_dict[cls]
        #         print(cur_art_list)
        #         cur_full_url_list = []
        #
        #         for art in cur_art_list:
        #             if art not in art_url_dict:
        #                 continue
        #
        #             if "latestnigeriannews" in art_url_dict[art] or "lapatriaenlinea" in art_url_dict[art]:
        #                 continue
        #
        #             print("art_url_dict[art]: ", art_url_dict[art])
        #             print("art_file_dict[art]: ", art_file_dict[art], "art_lineno_dict[art]", art_lineno_dict[art])
        #
        #             cur_full_url_list.append(art_url_dict[art])
        #
        #             try:
        #                 cur_art = load_article(art_file_dict[art], art_lineno_dict[art])
        #             except:
        #                 cur_art = {}
        #
        #
        #
        #         print("cur_full_url_list: ", cur_full_url_list)
        #
        #         # Path to the jsonl file
        #         cls_full_info_folder_path = "figs/cluster_validation/full_info/"
        #         if not os.path.exists(cls_full_info_folder_path):
        #             os.makedirs(cls_full_info_folder_path)
        #         cls_full_info_path = cls_full_info_folder_path + str(cls) + '_full_info.jsonl'
        #
        #         keys_to_include = ['url', 'title', "story_text", "language"]
        #         # Write the list to a jsonl file
        #         with jsonlines.open(cls_full_info_path, mode='w') as writer:
        #             # Write the list to a column
        #             for url in cur_full_url_list:
        #                 cur_save = {}
        #                 for key in keys_to_include:
        #                     if key in cur_art:
        #                         cur_save[key] = cur_art[key]
        #                 writer.write(cur_save)
        #
        #         print("cur_cls: ", cls)
        #         print("")
        #     except:
        #         traceback.print_exc()



    # check the temporal evolution of clusters, i.e. the top 3 clusters of each month
    if args.option == "oslom_temp":
        month_num = 6
        top_k = 5
        sample_size = 10000

        clustering_info_route = "network_data/input/oslom.tsv_oslo_files/partitions_level_0"

        month_dict = defaultdict(set)
        art_set = set()

        for pair in pairs.itertuples():
            cur_id1 = pair[4]
            cur_id2 = pair[10]

            cur_date1 = pair[3]
            cur_date2 = pair[9]

            month_dict[math.floor(cur_date1/30)].add(cur_id1)
            month_dict[math.floor(cur_date2/30)].add(cur_id2)

        '''6 month in total'''
        temporal_cls = defaultdict(lambda: defaultdict(set))
        with open(clustering_info_route, "r") as fh:
            line = fh.readline()
            cur_cls_num = 0
            while line:
                line = fh.readline()
                if len(line) <= 0:
                    continue
                if line[0] == '#':
                    cur_cls_num = int(line.split(" ")[1])
                    continue

                cur_cls = line.split(" ")
                for art_id in cur_cls:
                    if art_id == '\n':
                        continue
                    art_id = int(art_id)
                    art_set.add(art_id)
                    for cur_month in range(month_num):
                        if art_id in month_dict[cur_month]:
                            temporal_cls[cur_month][cur_cls_num].add(art_id)
                            break
        print(len(art_set), " articles in total....")

        # find the biggest 3 clusters in each month
        temporal_cls_size = defaultdict(lambda: defaultdict(int))
        for cur_month in temporal_cls:
            for cur_cls in temporal_cls[cur_month]:
                temporal_cls_size[cur_month][cur_cls] = len(temporal_cls[cur_month][cur_cls])
        top_temporal_cls = defaultdict(list)
        for cur_month in temporal_cls:
            cur_cls_size_rank = sorted(temporal_cls_size[cur_month].items(), key=lambda x: x[1], reverse=True)

            for i in range(top_k):
                cur_cls = cur_cls_size_rank[i][0]

                print(f"month {cur_month} {i}th top cluster size: {len(temporal_cls[cur_month][cur_cls])}")
                top_temporal_cls[cur_month].append(list(temporal_cls[cur_month][cur_cls])[:20])

        ''' retrieve the url of the samples from these top clusters '''

        print("start retrieve the url of the samples: ", datetime.now())

        art_dict = {}
        for lang in lang_list:
            print(f"processing {lang} at ", datetime.now())
            read_storyid_url_index(art_dict, f"indexes/{lang}-wiki-temp-v4.index")

        print("finish retrieve the url of the samples: ", datetime.now())

        top_cls_exp_dict = defaultdict(list)
        for cur_month in top_temporal_cls:
            for top_cls in top_temporal_cls[cur_month]:
                cls_sign = f"month{cur_month}_cls{top_cls}"
                for art in top_cls:
                    try:
                        print(art_dict[art])

                        file, lineno, url = art_dict[art]
                        top_cls_exp_dict["cluster"].append(cls_sign)
                        top_cls_exp_dict["url"].append(url)
                    except:
                        traceback.print_exc()
                # split sign between clusters
                top_cls_exp_dict["cluster"].append("")
                top_cls_exp_dict["url"].append("")

        top_cls_df = pd.DataFrame.from_dict(top_cls_exp_dict)
        top_cls_df.to_csv("network_data/temporal_cluster/temporal_cluster_exp.csv")


        ''' randomly select some cluster pairs to verify the average similarity '''
        with open("network_data/temporal_cluster/random_sample.json", "w") as output:
            for i in range(sample_size):
                rand_month = random.randint(0,month_num-1)
                rand_cls = random.sample(temporal_cls[rand_month].keys(), 1)[0]
                try:
                    art1_id, art2_id = random.sample(list(temporal_cls[rand_month][rand_cls]),2)

                    art1 = art_dict[art1_id]
                    art2 = art_dict[art2_id]

                    pair = {"art1":art1, "art2":art2}
                    output.write(json.dumps(pair))
                    output.write("\n")
                except:
                    traceback.print_exc()

    if args.option == "sim_cmp":
        within_cls_sample_size = 100000
        between_cls_sample_size = 10

        clustering_info_route = "network_data/input/oslom.tsv_oslo_files/partitions_level_0"


        pair_id_key = list(pairs_dict.keys())[0]
        pair_id_sim_dict = {}
        for i in range(len(pairs_dict[pair_id_key])):
            cur_pair_id = pairs_dict[pair_id_key][i]
            pair_id_sim_dict[cur_pair_id] = pairs_dict["similarity"][i]


        '''read cls info'''
        all_cls = defaultdict(set)
        with open(clustering_info_route, "r") as fh:
            line = fh.readline()
            cur_cls_num = 0
            while line:
                line = fh.readline()
                if len(line) <= 0:
                    continue
                if line[0] == '#':
                    cur_cls_num = int(line.split(" ")[1])
                    continue

                cur_cls = line.split(" ")
                if len(cur_cls) < min_cls_size:
                    continue
                for art_id in cur_cls:
                    if art_id == '\n':
                        continue
                    art_id = int(art_id)
                    all_cls[cur_cls_num].add(art_id)

        # within_start_time = datetime.now()
        # rand_within_cls_pair_sim_list = []
        # for i in range(within_cls_sample_size):
        #     rand_cls = random.sample(all_cls.keys(), 1)[0]
        #     try:
        #         art1_id, art2_id = random.sample(list(all_cls[rand_cls]),2)
        #
        #         rand_within_cls_pair = str(art1_id) + "_" + str(art2_id)
        #         if rand_within_cls_pair in pair_id_sim_dict:
        #             rand_within_cls_pair_sim_list.append(pair_id_sim_dict[rand_within_cls_pair])
        #
        #     except:
        #         traceback.print_exc()
        # within_end_time = datetime.now()
        # print(f"sampling {within_cls_sample_size} within cluster pairs takes ", within_end_time - within_start_time)

        # between_start_time = datetime.now()
        # rand_between_cls_pair_sim_list = []
        # for i in range(between_cls_sample_size):
        #     rand_cls1, rand_cls2 = random.sample(all_cls.keys(), 2)
        #     try:
        #         art1_id = random.sample(list(all_cls[rand_cls1]), 1)[0]
        #         art2_id = random.sample(list(all_cls[rand_cls2]), 1)[0]
        #
        #         rand_between_cls_pair = str(art1_id) + "_" + str(art2_id)
        #         if rand_between_cls_pair in pair_id_sim_dict:
        #             rand_between_cls_pair_sim_list.append(pair_id_sim_dict[rand_between_cls_pair])
        #
        #     except:
        #         traceback.print_exc()
        #
        # between_end_time = datetime.now()
        # print(f"sampling {between_cls_sample_size} between cluster pairs takes ", between_end_time - between_start_time)

        within_start_time = datetime.now()
        full_within_cls_pair_sim_list = []
        within_cls_pair_traverse_num = 0
        for cls in all_cls:
            for art1_id in all_cls[cls]:
                for art2_id in all_cls[cls]:
                    try:
                        cur_within_cls_pair = str(art1_id) + "_" + str(art2_id)
                        if cur_within_cls_pair in pair_id_sim_dict:
                            full_within_cls_pair_sim_list.append(pair_id_sim_dict[cur_within_cls_pair])
                    except:
                        traceback.print_exc()

                    within_cls_pair_traverse_num +=1
                    if within_cls_pair_traverse_num % 10000 == 0:
                        print(f"found ", len(full_within_cls_pair_sim_list), f" pairs within cluster pairs shared ne from traversing {within_cls_pair_traverse_num} within cluster pairs takes ", datetime.now() - within_start_time)
        within_end_time = datetime.now()
        print(f"sampling within cluster pairs takes ", within_end_time - within_start_time)

        # within_cls_shared_ne_aver_sim = np.mean(rand_within_cls_pair_sim_list)
        # between_cls_shared_ne_aver_sim = np.mean(rand_between_cls_pair_sim_list)

        within_cls_shared_ne_aver_sim = np.mean(full_within_cls_pair_sim_list)
        between_cls_shared_ne_aver_sim = (sum(pair_id_sim_dict.values())- sum(full_within_cls_pair_sim_list)) / (len(pair_id_sim_dict) - len(full_within_cls_pair_sim_list))

        print(len(full_within_cls_pair_sim_list)," pairs, average of all pairs sharing name entities and within same cluster:", within_cls_shared_ne_aver_sim)
        print(len(pair_id_sim_dict) - len(full_within_cls_pair_sim_list)," pairs, average of all pairs sharing name entities and between clusters:", between_cls_shared_ne_aver_sim)

    if args.option == "plt":
        # plot the different value of clusters
        fig = plt.figure(figsize=(13,10.4))

        value = [1, 1.96, 2.302, 2.80]
        name_list = ["Between", "Between \n sharing \n NE" , "Within \n shared \n no NE", "Within \n sharing \n NE"]
        color_list = ["silver", "grey", "lightcoral", 'r', 'blue']

        plt.bar(range(len(value)), value, tick_label=name_list, color=color_list)
        plt.xticks(size=24)
        plt.yticks(size=24)
        # plt.xlabel('Pair Kind', fontsize = 24)
        plt.ylabel('Average Similarity', fontsize = 32)
        # plt.title('Average similarity of different news pair kinds.')

        plt.savefig("figs/cluster_validation/aver_sim_cmp.png")


        fig = plt.figure(figsize=(13, 10.4))

        value = [1.56, 2.18, 2.83, 3.46]
        name_list = ["Very \n Dissimilar", "Somewhat \n Dissimilar", "Somewhat \n Similar", "Very \n Similar"]
        color_list = ["paleturquoise", "aquamarine", "cyan", 'blue']

        plt.bar(range(len(value)), value, tick_label=name_list, color=color_list)
        plt.xticks(size=24)
        plt.yticks(size=24)
        plt.xlabel('Pair Kind', fontsize=24)
        plt.ylabel('Average Similarity', fontsize=24)
        # plt.title('Average similarity of different news pair kinds.')

        plt.savefig("figs/cluster_validation/aver_label_sim_cmp.png")


        fig = plt.figure(figsize=(13, 10.4))

        value = [340, 40, 17, 180, 1250, 810, 220, 230]
        name_list = ["0~1", "1~2", "2~3", "3~7", "7~15", "15~30", "30~90", "90~180"]
        # color_list = ["silver", "grey", "lightcoral", 'r', 'blue']

        plt.bar(range(len(value)), value, tick_label=name_list)
        plt.xticks(size=23)
        plt.yticks(size=23)
        plt.xlabel('Cluster span (days)', fontsize=32)
        plt.ylabel('Cluster number', fontsize=32)

        plt.savefig("figs/cluster_validation/cls_date_class.png")

    if args.option == "cls_eval":
        cls_annotation = pd.read_csv("figs/cluster_validation/intruder_annotation.csv")
        inter_annotator_aggrement = irrCAC.raw.CAC(cls_annotation)
        print(inter_annotator_aggrement.gwet())