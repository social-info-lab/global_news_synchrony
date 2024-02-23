import re
import jieba
import sys
import json
import gzip


TEXT_MAX_LEN = 512

def truncatetext(txt, lang, length, tail_length):
    # print("good")

    head_length = length - tail_length


    if not isinstance(txt, str):
        txt = str(txt)

    txt=re.sub("[^\w ]","",txt) #only letters and spaces
    if lang=="zh":
        words = " ".join(jieba.cut(txt)).split()
    else:
        words = txt.split(" ")
    if len(words) > length:
        truncate = " ".join(words[:head_length]) + " " + " ".join(words[(-tail_length):])
    else:
        return " ".join(words)

    return truncate

def score_normalization(norm_type, score):
    if norm_type == "positive":
        return float((4-score)/3)
    elif norm_type == "unsigned":
        return float((5-2*score)/3)
    # error output
    return ""

def score_reverse_normalization(norm_type, normalized_score):
    if norm_type == "positive":
        return float(4-3*normalized_score)
    elif norm_type == "unsigned":
        return float((5-3*normalized_score)/2)
    # error output
    return ""

def load_article(file,lineno):
    try:
        lineno=int(lineno)
    except ValueError:
        print("DEBUG", lineno, "Note: known issue, needs debugging.")
        # probably some issue about saving/loading namedtuples
        sys.exit()

    file = file.replace(".gz", "").replace("home/scott/wikilinked", "work/xchen4_umass_edu/mediacloud_temp/scott/ner").replace("mnt/nfs/work1/grabowicz/xchen4/mediacloud_temp/scott/wikilinked" ,"work/xchen4_umass_edu/mediacloud_temp/scott/ner")

    print(file)
    # symbolic link doesn't look good with current inter-lang data storage format since the current wiki data is in scott's directory while the offsets are in my directory. Even use symbolic links it will always change the code since we use the json files and offsets at different time.
    with open(file.replace(".json", ".offsets"), "r") as fh:
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

