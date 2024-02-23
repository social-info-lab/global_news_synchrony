import re
import jieba


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

DISASTER_VOC = {
	"en":{"Natural disaster":["catastrophe","calamity", "hurricane","earthquake", "flood", "tornado", "tsunami","volcano", "wildfire", "landslide", "drought", "blizzard", "thunderstorm"],
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
		"Natural disaster": ["Katastrophe", "Unglück", "Hurrikan", "Erdbeben", "Überschwemmung", "Tornado", "Tsunami", "Vulkan", "Waldbrand", "Erdrutsch", "Dürre", "Schneesturm", "Gewitter"],
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
		"Natural disaster": ["Catástrofe", "Calamidad", "Huracán", "Terremoto", "Inundación", "Tornado", "Tsunami", "Volcán", "Incendio forestal", "Deslizamiento de tierra", "Sequía", "Ventisca", "Tormenta eléctrica"],
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
		"Natural disaster": ["Catastrophe naturelle", "Calamité", "Ouragan", "Séisme", "Inondation", "Tornade", "Tsunami", "Volcan", "Feu de forêt", "Glissement de terrain", "Sécheresse", "Blizzard", "Orage"],
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
		"Natural disaster": ["自然灾害", "大灾难", "飓风", "地震", "洪水", "龙卷风", "海啸", "火山", "野火", "滑坡", "干旱", "暴风雪", "雷暴"],
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
		"Natural disaster": ["كارثة طبيعية", "مصيبة", "إعصار", "زلزال", "فيضان", "زوبعة", "تسونامي", "بركان", "حرائق الغابات", "انهيار أرضي", "جفاف", "عاصفة ثلجية", "عاصفة رعدية"],
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
		"Natural disaster": ["felaket", "afet", "kasırga", "deprem", "sel", "hortum", "tsunami", "volkan", "orman yangını", "toprak kayması", "kuraklık", "tipi", "gök gürültülü fırtına"],
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
		"Natural disaster": ["catastrofe", "calamità", "uragano", "terremoto", "alluvione", "tornado", "tsunami", "vulcano", "incendio boschivo", "frana", "siccità", "bufera di neve", "temporale"],
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
		"Natural disaster": ["катастрофа", "бедствие", "ураган", "землетрясение", "наводнение", "торнадо", "цунами", "вулкан", "лесной пожар", "оползень", "засуха", "метель", "гроза"],
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