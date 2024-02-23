# this code is to match the weighted pairs inferenced by the deep learning model in "network_inference" section to the their countries and bias (if applied)
''''''
''' sbatch match_inference_country_bias_script.sh '''


import json
from datetime import datetime
import collections
import random
import os
import gc
import socket
from argparse import ArgumentParser
import numpy as np
import time
import sys
import gzip
from utils import LANG_FAMILY, unify_url
import pandas as pd
import glob
from collections import defaultdict
import geopy.distance


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-i", "--input-glob", dest="input_glob",
                            default="network_pairs/prediction/*/*", type=str,
                            help="Input globstring.")
    args = parser.parse_args()

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
    # country_outlets_url_pub_country = {}
    # country_outlets_url_about_country = {}
    country_outlets_id_pub_country = {}
    country_outlets_id_about_country = {}
    # for i in range(len(country_outlets_url)):
    #     country_outlets_url[i] = unify_url(country_outlets_url[i])
    #     country_outlets_url_pub_country[country_outlets_url[i]] = country_outlets_pub_country[i]
    #     country_outlets_url_about_country[country_outlets_url[i]] = country_outlets_about_country[i]
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
        country_alpha3_geography[country_alpha3_code[i]] = {"country_full_name":country_full_name[i], "country_latitude": country_latitude[i], "country_longitude": country_longitude[i]}

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
            bias_outlets_country[i] = bias_outlets_country[i].replace("United States","USA").split(" (")[0]
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

    data_path = args.input_glob
    n_processedpairs = 0
    n_lang_family_pairs = 0
    n_bias_pairs = 0
    n_bias_country_pairs = 0
    n_pub_country_pairs = 0
    n_about_country_pairs = 0
    n_with_country_pairs = 0
    n_with_geography_pairs = 0
    n_with_democracy_pairs = 0

    start_time = datetime.now()

    files=list(glob.glob(data_path))
    for file in files:
        output_file = file.replace("prediction","matched_prediction")
        output_dir = "/".join(output_file.split("/")[:-1])
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(output_file, "w") as outfile:
            with open(file, "r") as fh:
                for line in fh:
                    lang_family_symbol = 0
                    bias_symbol = 0
                    bias_country_symbol = 0
                    pub_country_symbol = 0
                    about_country_symbol = 0
                    country_geography_symbol = 0
                    country_democracy_symbol = 0

                    n_processedpairs += 1
                    if n_processedpairs % 1000 == 0:
                        print(f"processed {n_processedpairs} pairs...  {n_bias_pairs} pairs matched bias, {n_pub_country_pairs} pairs matched pub_country, {n_about_country_pairs} pairs matched about_country... ", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), flush=True)

                    pair = json.loads(line)
                    pair["media_url1"] = unify_url(pair["media_url1"])
                    pair["media_url2"] = unify_url(pair["media_url2"])

                    # match language family
                    try:
                        pair['lang_family1'] = LANG_FAMILY[pair["language1"]]
                    except:
                        pair['lang_family1'] = ""
                        lang_family_symbol = 1
                    try:
                        pair['lang_family2'] = LANG_FAMILY[pair["a2_language"]]
                    except:
                        pair['lang_family2'] = ""
                        lang_family_symbol = 1


                    '''match country according to media id'''
                    # match pub_country
                    try:
                        pair['pub_country1'] = country_outlets_id_pub_country[pair["media_id1"]]
                    except:
                        pair['pub_country1'] = ""
                        pub_country_symbol = 1
                    try:
                        pair['pub_country2'] = country_outlets_id_pub_country[pair["media_id2"]]
                    except:
                        pair['pub_country2'] = ""
                        pub_country_symbol = 1
                    # match about_country
                    try:
                        pair['about_country1'] = country_outlets_id_about_country[pair["media_id1"]]
                    except:
                        pair['about_country1'] = ""
                        about_country_symbol = 1
                    try:
                        pair['about_country2'] = country_outlets_id_about_country[pair["media_id2"]]
                    except:
                        pair['about_country2'] = ""
                        about_country_symbol = 1

                    # match bias
                    try:
                        pair['bias1'] = bias_outlets_url_bias[pair["media_url1"]]
                    except:
                        pair['bias1'] = ""
                        bias_symbol = 1
                    try:
                        pair['bias2'] = bias_outlets_url_bias[pair["media_url2"]]
                    except:
                        pair['bias2'] = ""
                        bias_symbol = 1
                    # match bias_country
                    try:
                        pair['bias_country1'] = bias_outlets_url_country[pair["media_url1"]]
                    except:
                        pair['bias_country1'] = ""
                        bias_country_symbol = 1
                    try:
                        pair['bias_country2'] = bias_outlets_url_country[pair["media_url2"]]
                    except:
                        pair['bias_country2'] = ""
                        bias_country_symbol = 1

                    # match democracy and geographic location based on country (pub_country or bias_country)

                    #first get the alpha3 code of the main country
                    if pair['pub_country1'] != "":
                        pair["main_country1"] = pair['pub_country1']
                    elif pair['bias_country1'] != "":
                        pair["main_country1"] = pair['bias_country1']
                    elif pair['about_country1'] != "":
                        pair["main_country1"] = pair['about_country1']
                    else:
                        pair["main_country1"] = ""

                    if pair['pub_country2'] != "":
                        pair["main_country2"] = pair['pub_country2']
                    elif pair['bias_country2'] != "":
                        pair["main_country2"] = pair['bias_country2']
                    elif pair['about_country2'] != "":
                        pair["main_country2"] = pair['about_country2']
                    else:
                        pair["main_country2"] = ""
                    
                    

                    # match to geographic location and democracy index
                    if pair["main_country1"] != "":
                        try:
                            pair["country_full_name1"] = country_alpha3_geography[pair["main_country1"]]["country_full_name"]
                            pair["latitude1"] = country_alpha3_geography[pair["main_country1"]]["country_latitude"]
                            pair["longitude1"] = country_alpha3_geography[pair["main_country1"]]["country_longitude"]
                        except:
                            country_geography_symbol = 1
                    if pair["main_country2"] != "":
                        try:
                            pair["country_full_name2"] = country_alpha3_geography[pair["main_country2"]]["country_full_name"]
                            pair["latitude2"] = country_alpha3_geography[pair["main_country2"]]["country_latitude"]
                            pair["longitude2"] = country_alpha3_geography[pair["main_country2"]]["country_longitude"]
                        except:
                            country_geography_symbol = 1
                    try:
                        # calculated as km
                        pair["geo_distance"] = geopy.distance.geodesic((pair["latitude1"], pair["longitude1"]), (pair["latitude2"], pair["longitude2"])).km
                    except:
                        country_geography_symbol = 1

                    try:
                        pair["democracy_index1"] = country_democracy_index[pair["country_full_name1"]]
                    except:
                        country_democracy_symbol = 1
                    try:
                        pair["democracy_index2"] = country_democracy_index[pair["country_full_name2"]]
                    except:
                        country_democracy_symbol = 1


                    if lang_family_symbol == 0:
                        n_lang_family_pairs += 1
                    if pub_country_symbol == 0:
                        n_pub_country_pairs += 1
                    if about_country_symbol == 0:
                        n_about_country_pairs += 1
                    if bias_symbol == 0:
                        n_bias_pairs += 1
                    if bias_country_symbol == 0:
                        n_bias_country_pairs += 1
                    if (bias_country_symbol == 0) or (pub_country_symbol == 0):
                        n_with_country_pairs += 1
                    if country_geography_symbol == 0:
                        n_with_geography_pairs += 1
                    if country_democracy_symbol == 0:
                        n_with_democracy_pairs += 1


                    outfile.write(json.dumps(pair))
                    outfile.write("\n")

    end_time = datetime.now()
    print(f"Finished at {end_time:%Y-%m-%d %H:%M:%S}")
    print(f"Took {end_time - start_time} seconds to processing {n_processedpairs} pairs....")
    print(f"{n_lang_family_pairs} matched language family pairs...")
    print(f"{n_pub_country_pairs} matched pub_country pairs, {n_about_country_pairs} matched about_country pairs, in total {n_with_country_pairs} pairs have country....")
    print(f"{n_with_geography_pairs} matched geographic location pairs...")
    print(f"{n_bias_pairs} matched bias pairs, {n_with_democracy_pairs} matched democracy pairs....")




    print()

    # unify_url(x["link"])