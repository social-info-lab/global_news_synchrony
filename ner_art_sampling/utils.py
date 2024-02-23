import re
import jieba
import sys
import datetime
import traceback
import collections
import numpy as np

# we log the git commit hashes used to generate news article pairs
import git



SHA = git.Repo(search_parent_directories=True).head.object.hexsha
# SHA = "xichen_test"
lang_list = ['en','de','es','pl','zh','fr','ar','tr','it','ru']
lang_full_name_list = ['English', 'German', 'Spanish', 'Polish', 'Chinese', 'French', 'Arabic', 'Turkish', 'Italian','Russian']
lang_dict = {key:1 for key in lang_list}
lang_full_name_dict = {key:1 for key in lang_full_name_list}

# the language family from Glottolog
LANG_FULL_NAME_MAP = {"English":"en",
                "German":"de",
                "Spanish":"es",
                "Polish":"pl",
                "French":"fr",
                "Chinese":"zh",
                "Arabic":"ar",
                "Turkish":"tr",
                "Italian":"it",
                "Russian":"ru",
                }

LANG_FAMILY = {"en":"Indo-European",
                "de":"Indo-European",
                "es":"Indo-European",
                "pl":"Indo-European",
                "fr":"Indo-European",
                "zh":"Sino-Tibetan",
                "ar":"Afro-Asiatic",
                "tr":"Turkic",
                "it":"Indo-European",
                "ru":"Indo-European",
                }

DISASTER_VOC = {
	"en":{"Natural disaster":["hurricane","earthquake", "flood", "tornado", "tsunami","volcano", "wildfire", "landslide", "drought", "blizzard", "thunderstorm"],
		"hurricane":["hurricane", "cyclone", "tropical", "landfall", "typhoon", "gale"],
		"earthquake":["earthquake", "epicenter","magnitude","aftershock","seismograph"],
		"flood":["flood","floodwater","rainfall","overflow", "floodplain", "inundation", "submerge", "waterlogged", "sandbag", "levee"],
		"tornado": ["tornado", "twister", "supercell", "Fujita","vortex"],
		"tsunami":["tsunami", "wave", "seismograph"],
		"volcano":["volcano", "magma", "crater", "ash", "fumarole", "lahar", "lava", "eruption"],
		"wildfire":["wildfire", "fire","burn", "firebreak", "firefighter", "ember", "conflagration", "firestorm"],
		"landslide":["landslide", "mudslide", "rockslide", "collapse", "subsidence"],
		"drought":["drought", "aquifer", "watershed", "desiccation", "desertification"],
		"blizzard":["blizzard", "snowstorm", "snowdrift", "snowfall", "frostbite","snowflake","avalanche", "snowpack","hailstones"],
		"thunderstorm":["thunderstorm", "thunderclap", "lightning", "thunderhead", "cumulonimbus"]
	},
	"de":{
		"Natural disaster": ["Hurrikan", "Erdbeben", "Überschwemmung", "Tornado", "Tsunami", "Vulkan", "Waldbrand", "Erdrutsch", "Dürre", "Schneesturm", "Gewitter"],
		"hurricane": ["Hurrikan", "Zyklon", "Tropen", "Landfall", "Taifun", "Sturm"],
		"earthquake": ["Erdbeben", "Epizentrum", "Magnitude", "Nachbeben", "Seismograf"],
		"flood": ["Überschwemmung", "Hochwasser", "Niederschlag", "Überlauf", "Überschwemmungsgebiet", "Überflutung", "Unterwasser", "Wassergetränkt", "Sandsack", "Deich"],
		"tornado": ["Tornado", "Wirbelsturm", "Superzelle", "Fujita", "Wirbel"],
		"tsunami": ["Tsunami", "Welle", "Seismograf"],
		"volcano": ["Vulkan", "Magma", "Krater", "Asche", "Fumarole", "Lahar", "Lava", "Ausbruch"],
		"wildfire": ["Waldbrand", "Feuer", "Verbrennung", "Feuerunterbrechung", "Feuerwehrmann", "Glut", "Großbrand", "Feuersturm"],
		"landslide": ["Erdrutsch", "Murgang", "Felssturz", "Einsturz", "Absenkung"],
		"drought": ["Dürre", "Grundwasserleiter", "Einzugsgebiet", "Austrocknung", "Wüstenbildung"],
		"blizzard": ["Schneesturm", "Schneetreiben", "Schneewehe", "Schneefall", "Erfrierung", "Schneeflocke", "Lawine", "Schneedecke", "Hagelkörner"],
		"thunderstorm": ["Gewitter", "Donnerschlag", "Blitz", "Donnerkopf", "Kumulonimbus"]
	},
	"es":{
		"Natural disaster": ["Huracán", "Terremoto", "Inundación", "Tornado", "Tsunami", "Volcán", "Incendio forestal", "Deslizamiento de tierra", "Sequía", "Ventisca", "Tormenta eléctrica"],
		"hurricane": ["Huracán", "Ciclón", "Tropical", "Desembarco", "Tifón", "Vendaval"],
		"earthquake": ["Terremoto", "Epicentro", "Magnitud", "Réplica", "Sismógrafo"],
		"flood": ["Inundación", "Aguas de inundación", "Lluvia", "Desbordamiento", "Llanura de inundación", "Inundación", "Sumergir", "Anegado", "Saco de arena", "Dique"],
		"tornado": ["Tornado", "Torbellino", "Supercelda", "Fujita", "Vórtice"],
		"tsunami": ["Tsunami", "Ola", "Sismógrafo"],
		"volcano": ["Volcán", "Magma", "Cráter", "Ceniza", "Fumarola", "Lahar", "Lava", "Erupción"],
		"wildfire": ["Incendio forestal", "Fuego", "Quemar", "Cortafuegos", "Bombero", "Brasa", "Conflagración", "Tormenta de fuego"],
		"landslide": ["Deslizamiento de tierra", "Alud de lodo", "Desprendimiento de rocas", "Colapso", "Hundimiento"],
		"drought": ["Sequía", "Acuífero", "Cuenca", "Desecación", "Desertificación"],
		"blizzard": ["Ventisca", "Tormenta de nieve", "Montón de nieve", "Nevada", "Congelación", "Copo de nieve", "Avalancha", "Capa de nieve", "Piedras de granizo"],
		"thunderstorm": ["Tormenta eléctrica", "Trueno", "Rayo", "Cabeza de trueno", "Cumulonimbus"]
	},
	"fr":{
		"Natural disaster": ["Ouragan", "Séisme", "Inondation", "Tornade", "Tsunami", "Volcan", "Feu de forêt", "Glissement de terrain", "Sécheresse", "Blizzard", "Orage"],
		"hurricane": ["Ouragan", "Cyclone", "Tropical", "Atterrissage", "Typhon", "Coup de vent"],
		"earthquake": ["Séisme", "Épicentre", "Magnitude", "Réplique", "Sismographe"],
		"flood": ["Inondation", "Eaux d'inondation", "Précipitations", "Débordement", "Plaine inondable", "Submersion", "Submergé", "Gorgé d'eau", "Sac de sable", "Digue"],
		"tornado": ["Tornade", "Tourbillon", "Supercellule", "Fujita", "Vortex"],
		"tsunami": ["Tsunami", "Vague", "Sismographe"],
		"volcano": ["Volcan", "Magma", "Cratère", "Cendre", "Fumerolle", "Lahar", "Lave", "Éruption"],
		"wildfire": ["Feu de forêt", "Feu", "Brûler", "Coupe-feu", "Pompiers", "Braise", "Conflagration", "Tempête de feu"],
		"landslide": ["Glissement de terrain", "Coulée de boue", "Éboulement", "Effondrement", "Affaissement"],
		"drought": ["Sécheresse", "Aquifère", "Bassin versant", "Dessiccation", "Désertification"],
		"blizzard": ["Blizzard", "Tempête de neige", "Congère", "Chute de neige", "Engelure", "Flocon de neige", "Avalanche", "Manteau neigeux", "Grêlons"],
		"thunderstorm": ["Orage", "Coup de tonnerre", "Foudre", "Tête de tonnerre", "Cumulonimbus"]
	},
	"zh":{
		"Natural disaster": ["飓风", "地震", "洪水", "龙卷风", "海啸", "火山", "野火", "滑坡", "干旱", "暴风雪", "雷暴"],
		"hurricane": ["飓风", "气旋", "热带", "登陆", "台风", "大风"],
		"earthquake": ["地震", "震中", "震级", "余震", "地震仪"],
		"flood": ["洪水", "洪水", "降雨量", "溢出", "洪泛区", "淹没", "淹水", "浸水", "沙袋", "堤坝"],
		"tornado": ["龙卷风", "旋风", "超级胞", "藤田级数", "涡旋"],
		"tsunami": ["海啸", "波浪", "地震仪"],
		"volcano": ["火山", "岩浆", "火山口", "灰烬", "气孔", "熔岩流", "熔岩", "爆发"],
		"wildfire": ["野火", "火", "燃烧", "火隙", "消防员", "余烬", "大火", "火风暴"],
		"landslide": ["滑坡", "泥石流", "岩石滑坡", "坍塌", "下沉"],
		"drought": ["干旱", "含水层", "流域", "干燥", "沙漠化"],
		"blizzard": ["暴风雪", "雪暴", "雪堆", "降雪", "冻伤", "雪花", "雪崩", "积雪", "冰雹"],
		"thunderstorm": ["雷暴", "雷声", "闪电", "雷云", "积雨云"]
	},
	"ar":{
		"Natural disaster": ["إعصار", "زلزال", "فيضان", "زوبعة", "تسونامي", "بركان", "حرائق الغابات", "انهيار أرضي", "جفاف", "عاصفة ثلجية", "عاصفة رعدية"],
		"hurricane": ["إعصار", "إعصار", "استوائي", "الوصول إلى اليابسة", "تايفون", "عاصفة"],
		"earthquake": ["زلزال", "مركز الزلزال", "قوة", "هزة ساقطة", "سجل زلزالي"],
		"flood": ["فيضان", "مياه الفيضان", "هطول الأمطار", "ت overflowدفق", "سهل الفيضان", "غمر", "غمر", "مشبع بالماء", "كيس رمل", "سد"],
		"tornado": ["زوبعة", "زوبعة", "خلية عاصفة", "فوجيتا", "دوامة"],
		"tsunami": ["تسونامي", "موجة", "سجل زلزالي"],
		"volcano": ["بركان", "ماجما", "حفرة", "رماد", "فومارول", "لاهار", "حمم", "ثورة بركانية"],
		"wildfire": ["حريق غابات", "حريق", "حرق", "حاجز حريق", "رجل إطفاء", "جمرة", "حريق كبير", "عاصفة نارية"],
		"landslide": ["انهيار أرضي", "انزلاق طيني", "انزلاق صخري", "انهيار", "تراجع"],
		"drought": ["جفاف", "خزان مائي تحت الأرض", "حوض مائي", "تجفيف", "تصحر"],
		"blizzard": ["عاصفة ثلجية", "عاصفة ثلجية", "تراكم ثلجي", "تساقط الثلوج", "تجمد", "ندفة ثلج", "انهيار ثلجي", "تراكم ثلجي", "حجارة البرد"],
		"thunderstorm": ["عاصفة رعدية", "صوت الرعد", "برق", "رأس العاصفة", "سحابة ركامية"]
	},
	"tr":{
		"Natural disaster": ["kasırga", "deprem", "sel", "hortum", "tsunami", "volkan", "orman yangını", "toprak kayması", "kuraklık", "tipi", "gök gürültülü fırtına"],
		"hurricane": ["kasırga", "siklon", "tropikal", "karaya ulaşma", "tifon", "fırtına"],
		"earthquake": ["deprem", "episantr", "büyüklük", "artçı sarsıntı", "sismograf"],
		"flood": ["sel", "sel suyu", "yağış", "taşma", "sel alanı", "suların yükselmesi", "suya batmak", "su basması", "kum torbası", "bent"],
		"tornado": ["hortum", "hortum", "süper hücre", "Fujita", "girdap"],
		"tsunami": ["tsunami", "dalga", "sismograf"],
		"volcano": ["volkan", "magma", "krater", "kül", "fumarol", "lahar", "lava", "püskürme"],
		"wildfire": ["orman yangını", "yangın", "yanma", "yangın kesme", "itfaiyeci", "kor", "büyük yangın", "aşırı yangın"],
		"landslide": ["toprak kayması", "çamur kayması", "kaya düşmesi", "çökme", "çöküş"],
		"drought": ["kuraklık", "yer altı su tablası", "havza", "kuruma", "çölleşme"],
		"blizzard": ["tipi", "kar fırtınası", "kar yığını", "kar yağışı", "donma", "kar tanesi", "çığ", "kar tabakası", "dolu taşı"],
		"thunderstorm": ["gök gürültülü fırtına", "şimşek çakması", "yıldırım", "fırtına bulutu", "kümülonimbus"]
	},
	"it":{
		"Natural disaster": ["uragano", "terremoto", "alluvione", "tornado", "tsunami", "vulcano", "incendio boschivo", "frana", "siccità", "bufera di neve", "temporale"],
		"hurricane": ["uragano", "ciclone", "tropicale", "atterraggio", "tifone", "burrasca"],
		"earthquake": ["terremoto", "epicentro", "magnitudo", "scossa di assestamento", "sismografo"],
		"flood": ["alluvione", "acqua d'inondazione", "pioggia", "straripamento", "pianura alluvionale", "inondazione", "sommersione", "saturazione d'acqua", "sacco di sabbia", "argine"],
		"tornado": ["tornado", "turbine", "supercella", "Fujita", "vortice"],
		"tsunami": ["tsunami", "onda", "sismografo"],
		"volcano": ["vulcano", "magma", "cratere", "cenere", "fumarola", "lahar", "lava", "eruzione"],
		"wildfire": ["incendio boschivo", "fuoco", "bruciare", "interruzione del fuoco", "pompiere", "braci", "incendio", "tempesta di fuoco"],
		"landslide": ["frana", "frana di fango", "valanga di rocce", "crollo", "sprofondamento"],
		"drought": ["siccità", "falda acquifera", "bacino idrografico", "essiccazione", "desertificazione"],
		"blizzard": ["bufera di neve", "tempesta di neve", "banchisa", "nevicate", "congelamento", "fiocco di neve", "valanga", "manto nevoso", "grandine"],
		"thunderstorm": ["temporale", "tuono", "fulmine", "nube temporalesca", "cumulonembo"]
	},
	"ru":{
		"Natural disaster": ["ураган", "землетрясение", "наводнение", "торнадо", "цунами", "вулкан", "лесной пожар", "оползень", "засуха", "метель", "гроза"],
		"hurricane": ["ураган", "циклон", "тропический", "высадка на берег", "таифун", "шторм"],
		"earthquake": ["землетрясение", "эпицентр", "магнитуда", "подземные толчки", "сейсмограф"],
		"flood": ["наводнение", "потоп", "осадки", "перелив", "пойма", "затопление", "погружение", "затопленный", "мешок с песком", "дамба"],
		"tornado": ["торнадо", "смерч", "суперклетка", "Фуджита", "вихрь"],
		"tsunami": ["цунами", "волна", "сейсмограф"],
		"volcano": ["вулкан", "магма", "кратер", "пепел", "фумарола", "лахар", "лава", "извержение"],
		"wildfire": ["лесной пожар", "огонь", "горение", "противопожарная полоса", "пожарник", "тлеющий уголь", "конфлаграция", "пожарная буря"],
		"landslide": ["оползень", "грязевой поток", "каменный обвал", "крушение", "осадка"],
		"drought": ["засуха", "водоносный слой", "водосбор", "высыхание", "опустынивание"],
		"blizzard": ["метель", "снежная буря", "сугроб", "снегопад", "обморожение", "снежинка", "лавина", "снежный покров", "градины"],
		"thunderstorm": ["гроза", "гром", "молния", "облако-грозовик", "кумулонимбус"]
	}

	}

News = collections.namedtuple("News", "file, lineno, reladate, vec") # reladate is the number of days from 2020-01-01 to the published date of the article
Biased_News = collections.namedtuple("Biased_News", "outlet_flag, match_country, mbfc_match_bias, file, lineno, reladate, vec")
blockwords=["sport","reddit.com","facebook.com","twitter.com","facebook.com","reddit.com","fb.com","wikipedia.org","epochtimes.com","youtube.com", "slideshare.net"]
# so this is to say, if an outlet is shown as "center" in any one of the collections, we regard it as "center", otherwise we take its most left bias among the collections.
bias_class = ["center", "left", "left_center", "right_center", "right", "conspiracy", "fake_news", "pro_science", "satire"]
valid_bias_class = ["center", "left", "left_center", "right_center", "right"]
strong_bias_class = ["left", "right"]
democracy_index_class = [0,2, 4, 6, 8, 10]
distance_class = [500, 1000, 2000, 3000, 4000, 5000, 10000, 20000]
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'July'] # the month time of our data
month_len = 30

#politic groups
# political_group_list = ["nato","eunion","brics","csto"]
political_group_list = ["nato","eunion","brics"]
continent_list = ["Asia", "Europe", "Africa", "Oceania", "North America", "South America", "Antarctica"]

MIN_ARTICLE_LENGTH = 100 # Articles less than 100 *words* are discarded (vary according to translation ratio across languages)
# MIN_NE_SIM = {'intra-lang':0.35, 'inter-lang':0.4} # for sampling, Article pairs with less than 0.2 (1/3 intersection) name entity similarity are discarded
# MIN_NE_SIM = {'intra-lang':0.1, 'inter-lang':0.1}
MIN_NE_SIM = {'intra-lang':0.3, 'inter-lang':0.2} # for network inference
# MAX_TXT_SIM = 0.25 # for sampling, Article pairs with more than 0.25 text similarity (40% intersection) are regarded as duplicates
MAX_TXT_SIM = 0.25 # for network inference
MIN_SENTENCE_LENGTH = 40 # Article sentences less than 40 characters are not considered as a a valid sentences
MAX_SENTENCE_DUP_NUM = 0 # Article pairs with more than 1 shared sentences will be regarded as duplicate suspects
MIN_NE_NUM = 5
MIN_SHARED_TF_IDF_SUM = 1.25
CANDIDATE_NUM = 50 # candidate number for each pair
DATE_WINDOW = 5
decay_factor = [0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1] # the first one '0' is just to unify the serial numbers here and the numbers in the classifier section
START_DATE = '2020-01-01'

GEN_RECORD_FILENAME = "indexes/gen_record.txt"
REFINE_RECORD_FILENAME = "indexes/refine_record.txt"

# transform the meta news article silarity for more intuitive comparison
def trans_node_sim(sim):
	return (pow(300, 1/(sim + 0.1) ) )

def trans_edge_sim(sim):
	return (pow(1000, 1/(sim + 0.1) ) )

def trans_dem_idx_sim(sim):
	return (pow(100, 1/(sim + 0.1) ) )

# url matching, remove http prefix, refine the format of special character
def unify_url(url):
    try:
        unified_url = url.strip().replace("https://", "").replace("http://", "").replace("www.","")

        # remove the last "/"
        if unified_url[-1] == "/":
            unified_url = unified_url[:-1]

        if "?" in unified_url:
            split_url = unified_url.split('/')
            # usually "?" symbol is in the last part of the url, remove it
            unified_url = "".join(split_url[:-1])

        SpecialCharList = ["+", "*"]
        for eachSpecialChar in SpecialCharList:
            escapedChar = "\%s" % eachSpecialChar  # '\\.'
            unified_url = unified_url.replace(eachSpecialChar, escapedChar)
        return unified_url
    except:
        traceback.print_exc()
        return ""




# split the text to tokens
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

# split the text to hashed sentences
def text2sentence(text, lang):
	if lang == "zh":
		sentences = text.split("。")
	else:
		sentences = text.split(".")
	sentence_hashes = set()
	for sentence in sentences:
		if len(sentence) > MIN_SENTENCE_LENGTH and sentence != "" and sentence != " ":
			sentence_hashes.add(hash(sentence.strip()))
	return sentence_hashes


# get the NE type ('spacy' or 'polyglot') according to the specific language
def name_entity_type(lang):
	real_lang = lang.split('_')[0]

	if real_lang in ['ko',  'ar', 'tr', 'ko_test', 'ar_test', 'tr_test']:
		return 'polyglot'
	elif real_lang in ['en', 'es', 'fr', 'zh', 'pl', 'pt', 'de', 'ru', 'en_test', 'es_test', 'fr_test', 'zh_test', 'pl_test', 'pt_test', 'de_test', 'ru_test']:
		return 'spacy'
	return 'wrong language'

def translation_ratio(lang):
	# ratio for zh and ko here account for the words extract by jieba rather than the original text length
	ratio = {'en': 1, 'pl': 0.86, 'zh': 1, 'de': 0.87, 'es': 1.13, 'fr': 1.1, 'ko': 1, 'pt': 0.99, 'ar': 0.9, 'ru': 0.89, 'tr': 1, 'it': 1}

	real_lang = lang.split('_')[0]

	if real_lang in ratio:
		return ratio[real_lang]

	print("we don't cover this language")
	return 1

#compute the maximum number of name entities in the article
def max_NE(art):
	max_num = 0
	if 'spacy' in art and len(art['spacy']) > max_num:
		max_num = len(art['spacy'])
	if 'polyglot' in art and len(art['polyglot']) > max_num:
		max_num = len(art['polyglot'])
	if 'wiki_concepts' in art and len(art['wiki_concepts']) > max_num:
		max_num = len(art['wiki_concepts'])
	return max_num

# count the total line number of a file
def file_line_num(file):
	count = -1
	for count, line in enumerate(open(file, 'rU')):
		pass
	count += 1

	return count

# get the memory usage of an object
def mem_usage(obj):
	usage_mb = sys.getsizeof(obj) / 1024 ** 2 # convert bytes to megabytes
	return "{:03.2f} MB".format(usage_mb)

# locate the date is in which week of the month, default is today
def locate_week(date_str=None):
	if date_str and isinstance(date_str, str):
		now_time = datetime.datetime.strptime(date_str + " 00:00:00", "%Y-%m-%d %H:%M:%S")
	else:
		now_time = datetime.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
	# 当月第一天
	one_time = now_time.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

	year_num = int(now_time.strftime('%Y'))

	month_num = int(now_time.strftime('%m'))

	week_num = int(now_time.strftime('%W')) - int(one_time.strftime('%W')) + 1

	return year_num, month_num, week_num

def date_diff(date1_str, date2_str):
	date_set1 = date1_str.split("-")
	date_set2 = date2_str.split("-")
	date1 = datetime.datetime(int(date_set1[0]), int(date_set1[1]), int(date_set1[2]))
	date2 = datetime.datetime(int(date_set2[0]), int(date_set2[1]), int(date_set2[2]))

	return np.abs(date2 - date1)

# take art1.vec and art2.vect as input, convert them into sorted list, and then traverse once
# return both the jaccard similarity and the length of the union set of the two lists
def jaccard_similarity(vec1, vec2):
	# vec1 and vec2 are sorted tuples, so we don't need this any more
	# sorted_list1 = list(vec1)
	# sorted_list1.sort()
	# sorted_list2 = list(vec2)
	# sorted_list2.sort()
	# length1, length2 = len(sorted_list1), len(sorted_list2)
	length1, length2 = len(vec1), len(vec2)
	# count the sum of the numbers of duplicates in both lists
	repeat_times = 0
	# there are no duplicates, because we use Counter in `create_index`
	# for i in range(0,length1 - 1):
	# 	if sorted_list1[i+1] == sorted_list1[i]:
	# 		repeat_times += 1
	# for i in range(0, length2 - 1):
	# 	if sorted_list2[i+1] == sorted_list2[i]:
	# 		repeat_times += 1
	intersection = list()
	index1 = index2 = 0
	while index1 < length1 and index2 < length2:
		# num1 = sorted_list1[index1]
		# num2 = sorted_list2[index2]
		num1 = vec1[index1]
		num2 = vec2[index2]
		if num1 == num2:
			# ensure we only count a unique element once
			if not intersection or num1 != intersection[-1]:
				intersection.append(num1)
			index1 += 1
			index2 += 1
		elif num1 < num2:
			index1 += 1
		else:
			index2 += 1

	len_intersection = len(intersection)
	len_union = length1 + length2 - repeat_times - len_intersection
	if len_union:
		return len_intersection/len_union, len_union
	return 0, 0

def cosine_similarity(list1, list2, intersect_list):
	cosine_up_term = 0
	ne1_sum_sqrt = 0
	ne2_sum_sqrt = 0

	if intersect_list == []:
		return 0

	for ne1 in list1:
		if ne1[0] in intersect_list:
			for ne2 in list2:
				if ne1[0] == ne2[0]:
					cosine_up_term += ne1[1] * ne2[1]
					break

	for ne1 in list1:
		ne1_sum_sqrt += ne1[1] * ne1[1]
	for ne2 in list2:
		ne2_sum_sqrt += ne2[1] * ne2[1]
	cosine_bot_term = np.sqrt(ne1_sum_sqrt) * np.sqrt(ne2_sum_sqrt)

	if cosine_bot_term == 0:
		return 0
	else:
		return cosine_up_term/cosine_bot_term

# repeat_type is chosen from ["non_repeat", "repeat"]
def compute_pairwise_tf_idf(ne_list1, ne_list2, idf_dict, repeat_type):
	tf_idf_dict = collections.defaultdict(float)
	intersect_set, union_dict = compute_shared_ne(ne_list1, ne_list2, repeat_type)

	for ne_name in list(union_dict):
		if ne_name in idf_dict:
			tf = union_dict[ne_name] / sum(union_dict.values())
			tf_idf_dict[ne_name] = tf * idf_dict[ne_name]

	return intersect_set, tf_idf_dict

def compute_classic_tf_idf(ne_list, idf_dict):
	tf_idf_dict = collections.defaultdict(float)
	ne_count_sum = 0
	for ne in ne_list:
		ne_count_sum += ne[1]

	for ne in ne_list:
		ne_name = ne[0]
		ne_count = ne[1]
		if ne_name in idf_dict:
			tf = ne_count / ne_count_sum
			tf_idf_dict[ne_name] = tf * idf_dict[ne_name]

	tf_idf_list = []
	for ne in tf_idf_dict:
		tf_idf_list.append((ne, tf_idf_dict[ne]))

	return tf_idf_list

def compute_pairwise_bm25(ne_list1, ne_list2, idf_dict, art1_len, art2_len, aver_art_len, repeat_type):
	# according to classic usual
	k1 = 1.2
	b = 0.75

	bm25_dict = collections.defaultdict(float)
	intersect_set, union_dict = compute_shared_ne(ne_list1, ne_list2, repeat_type)

	for ne_name in list(union_dict):
		if ne_name in idf_dict:
			tf = union_dict[ne_name] / sum(union_dict.values())
			bm25_dict[ne_name] = tf * (k1+1) / (tf + k1*(1-b+b*((art1_len + art2_len)/2)/aver_art_len)) * idf_dict[ne_name]

	# print("ne_list1:", ne_list1, "ne_list2:", ne_list2)
	# print("intersect_set:", intersect_set)
	# print("bm25_dict:", bm25_dict)
	print()

	return intersect_set, bm25_dict

def compute_bm25(ne_list, idf_dict, art_len, aver_art_len):
	# according to classic usual
	k1 = 1.2
	b = 0.75

	bm25_dict = collections.defaultdict(float)
	ne_count_sum = 0
	for ne in ne_list:
		ne_count_sum += ne[1]

	for ne in ne_list:
		ne_name = ne[0]
		ne_count = ne[1]
		if ne_name in idf_dict:
			tf = ne_count / ne_count_sum
			bm25_dict[ne_name] = tf * (k1+1) / (tf + k1*(1-b+b*art_len/aver_art_len)) * idf_dict[ne_name]

	bm25_list = []
	for ne in bm25_dict:
		bm25_list.append((ne, bm25_dict[ne]))

	return bm25_list

def compute_shared_ne(vec1, vec2, repeat_type):
	# here we compute the sum of the shared ne counts
	intersect_list = []
	union_dict = collections.defaultdict(int)

	if repeat_type == "non_repeat":
		for each in vec1:
			union_dict[each[0]] = 1
		for each in vec2:
			union_dict[each[0]] = 1
	elif repeat_type == "repeat":
		for each in vec1:
			union_dict[each[0]] += each[1]
		for each in vec2:
			union_dict[each[0]] += each[1]
	else:
		print("wrong repeat type...")
		return [], union_dict

	for each1 in vec1:
		for each2 in vec2:
			if each1[0] == each2[0]:
				intersect_list.append(each1[0])
				break

	# for test
	# print(intersect_list)

	shared_ne_count = 0
	for intersect_ne in intersect_list:
		shared_ne_count += union_dict[intersect_ne]

	return intersect_list, union_dict

def count_shared_sum(intersect_list, union_dict):
	shared_sum = 0
	for intersect_ne in intersect_list:
		shared_sum += union_dict[intersect_ne]
	return shared_sum

def list_to_counter_tuple_list(input_list, repeat_type):
	counter = collections.Counter(input_list)
	tuple_list = []

	for key in counter:
		if repeat_type == "non_repeat":
			tuple_list.append((key, 1))
		elif repeat_type == "repeat":
			tuple_list.append((key, counter[key]))
	return tuple_list

def match_to_distance_class(dist, distance_class):
	class_len = len(distance_class)

	for i in range(class_len):
		if dist <= distance_class[i]:
			return i
	return class_len