from collections import defaultdict
import time
import sys

import pandas as pd



from pyecharts import options as opts
from pyecharts.charts import Geo
from pyecharts.globals import ChartType, SymbolType, GeoType

geo = Geo()

#Add coordinate points, add names and longitude and latitude
geo.add_coordinate(name="China",longitude=104.195,latitude=35.675)
geo.add_coordinate(name="Australia",longitude=133.775,latitude=-25.274)
geo.add_coordinate(name="Brazil",longitude=-51.925,latitude=-14.235)
geo.add_coordinate(name="South Africa",longitude=22.937,latitude=-30.559)
geo.add_coordinate(name="India",longitude=78.962,latitude=20.593)
geo.add_coordinate(name="Peru",longitude=-75.015,latitude=-9.189)
geo.add_coordinate(name="Iran",longitude=53.688,latitude=32.427)
geo.add_coordinate(name="Ukraine",longitude=31.165,latitude=48.379)
geo.add_coordinate(name="Canada",longitude=-106.346,latitude=56.130)
geo.add_coordinate(name="Mongolia",longitude=103.847,latitude=46.862)
geo.add_coordinate(name="Russia",longitude=37.618,latitude=55.751)
geo.add_coordinate(name="Mauritania",longitude=21.008,latitude=-10.941)
geo.add_coordinate(name="Kazakhstan",longitude=66.924,latitude=48.019)
geo.add_coordinate(name="UAE",longitude=53.848,latitude=23.424)
geo.add_coordinate(name="Malaysia",longitude=101.976,latitude=4.210)
geo.add_coordinate(name="New Zealand",longitude=174.886,latitude=-40.900)
geo.add_coordinate(name="Indonesia",longitude=113.921,latitude=-0.789)
geo.add_coordinate(name="Sweden",longitude=18.643,latitude=60.128)
geo.add_coordinate(name="Mexico",longitude=-102.553,latitude=23.634)
geo.add_coordinate(name="Sierra Leone",longitude=-11.779,latitude=8.461)

#Add data item
geo.add_schema(maptype="world")
geo.add("",[("Australia",128326),
      ("Brazil",44037),
      ("South Africa",7649),
      ("India",3562),
      ("Peru",2779),
      ("Iran",2698),
      ("Ukrainie",2040),
      ("Canada",1792),
      ("Mongolia",1514),
      ("Russia",1069),
      ("Mauritania",1374),
      ("Kazakhsan",701),
      ("UAE",490),
      ("Malaysia",554),
      ("New Zealand",422),
      ("Indonesia",148),
      ("Sweden",113),
      ("Mexico",121),
      ("Sierra Leone",109),
      ],type_=ChartType.EFFECT_SCATTER)

#Draw flow direction
geo.add("flow chart", [
  ("Australia","China"),
  ("Brazil","China"),
  ("South Africa","China"),
  ("India","China"),
  ("Peru","China"),
  ("Iran","China"),
  ("Ukraine","China"),
  ("Canada","China"),
  ("Mongolia","China"),
  ("Russia","China"),
  ("Mauritania","China"),
  ("Kazakhstan","China"),
  ("UAE","China"),
  ("Malaysia","China"),
  ("New Zealand","China"),
  ("Indonesia","China"),
  ("Sweden","China"),
  ("Mexico","China"),
  ("Sierra Leone","China"),
      ],
    type_= GeoType.LINES,
   effect_opts=opts.EffectOpts(symbol=SymbolType.ARROW,symbol_size=5,color="yellow"),
    linestyle_opts=opts.LineStyleOpts(curve=0.2),)

geo.set_series_opts(label_opts=opts.LabelOpts(is_show=False))
geo.set_global_opts(visualmap_opts=opts.VisualMapOpts(max_=130000),title_opts=opts.TitleOpts(title="mygeo"))
geo.render("geo_lines.html")





k = 50
ne_stat = {}
ne_map = {}
search = []
link_count = 0

a = pd.read_csv("/Users/xichen/Documents/GitHub/mediacloud/ner_art_sampling/bias_dataset/V-Dem/Country_Year_V-Dem_Full+others_CSV/V-Dem-CY-Full+Others-v12.csv")

with open("/Users/xichen/Documents/GitHub/mediacloud/ner_art_sampling/top10-ne-art-wiki-filtered.index", "r") as fh:
    line_num = 0
    cur_pair_num = 0
    for line in fh:
        line_num += 1
        if line_num % 2 == 1:
            cur_stat = line.replace("\n","").split(" ")
            cur_stat[0] = int(cur_stat[0])
            cur_stat[1] = int(cur_stat[1])
            ne_stat[cur_stat[0]] = cur_stat[1]
        print(line_num)

L = sorted(ne_stat.items(), key=lambda item: item[1], reverse=True)

for item in L[:k]:
    search.append(item[0])
    link_count += item[1] * (item[1] - 1)

print(L[:k])
print("link_count:", link_count)



with open("/Users/xichen/Documents/GitHub/mediacloud/ner_art_sampling/en-wiki-v2_ne.index", "r") as fh:
    line_num = 0
    cur_pair_num = 0
    for line in fh:
        cur_map = line.replace("\n", "").split("\t")
        cur_map[1] = int(cur_map[1])
        if cur_map[1] in search:
            ne_map[cur_map[0]] = cur_map[1]



with open("/Users/xichen/Documents/GitHub/mediacloud/ner_art_sampling/top10-ne-art-wiki-filtered.index", "r") as fh:
    line_num = 0
    cur_pair_num = 0
    for line in fh:
        line_num += 1
        if line_num % 2 == 0:
            cur_stat = line.replace("\n","").split(" ")
            cur_stat[0] = int(cur_stat[0])
            cur_stat[1] = int(cur_stat[1])
            ne_stat[cur_stat[0]] = cur_stat[1]
        print(line_num)


print()
# for k,v in ne_map.items():
#     print(f"{k}:{v}")



'''compare the efficiency and memory of set and dict'''
# r = 1000
# a = set()
# b = defaultdict(set)
# c1 = 1986
# c2 = 1745
# c = r * c1 + c2
#
# for i in range(r):
#     for j in range(r):
#         a.add(i * r + j)
#         b[i].add(j)
#
# a_memory = sys.getsizeof(a) / 1024 / 1024
# b_memory = sys.getsizeof(a) / 1024 / 1024
#
# T1 = time.time()
# if c not in a:
#     a.add(c)
# T2 = time.time()
# print('set运行时间:%s毫秒' % ((T2 - T1) * 1000), "    current memory use: ", a_memory, "MB")
#
# T1 = time.time()
# if c2 not in b[c1]:
#     b[c1].add(c2)
# T2 = time.time()
# print('dict运行时间:%s毫秒' % ((T2 - T1) * 1000), "    current memory use: ", b_memory, "MB")