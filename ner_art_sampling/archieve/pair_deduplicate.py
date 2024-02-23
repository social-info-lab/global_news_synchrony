from argparse import ArgumentParser
import sys
import datetime

def cal_digit(a):
    c = 0
    while a!=0:
        a=int(a/10)
        c += 1
    return c


lang_list = ['en','de','es','pl','zh','fr','ar','tr','it','ru']

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-i", "--input-filename", dest="input_filename",
                        default="indexes/update-ne-art-wiki.index", type=str,
                        help="Index filename.")
    parser.add_argument("-o", "--output-filename", dest="output_filename",
                        default="indexes/pair-deduplicate-wiki.index", type=str,
                        help="Output pair filename.")
    args = parser.parse_args()

    # attach this to the end of id for saving memory (just need 1 digit instead of 2 digits to store 10 language)
    lang_dict = {'en':0,'de':1,'es':2,'pl':3,'zh':4,'fr':5,'ar':6,'tr':7,'it':8,'ru':9}
    pair_set = set()

    with open(args.input_filename, "r") as fh:
        line_num = 0
        cur_pair_num = 0
        for line in fh:
            line_num += 1
            if line_num % 2 == 0:
                cur_arts = eval(line)
                for art1 in cur_arts:
                    art1_serial = int(art1[1])
                    art1_digit = cal_digit(art1_serial)
                    lang1_ind = lang_dict[art1[0].split("/")[-1][:2]]
                    for art2 in cur_arts:
                        art2_serial = int(art2[1])
                        # skip same article within a pair
                        if art1_serial != art2_serial or art1[0] != art2[0]:
                            art2_digit = cal_digit(art2_serial)
                            lang2_ind = lang_dict[art2[0].split("/")[-1][:2]]

                            if art1_serial <= art2_serial:
                                encode_pair_key = (1000*art1_digit + 100*art2_digit + 10*lang1_ind + lang2_ind) * (10**(art1_digit + art2_digit)) + art1_serial * (10**art2_digit) + art2_serial
                            else:
                                encode_pair_key = (1000*art2_digit + 100*art1_digit + 10*lang2_ind + lang1_ind) * (10**(art2_digit + art1_digit)) + art2_serial * (10**art1_digit) + art1_serial
                            pair_set.add(encode_pair_key)
                            cur_pair_num += 1

                            if cur_pair_num % 10 == 0:
                                cur_pair_set_memory = sys.getsizeof(pair_set)/1024/1024
                                print(f"processing {line_num} lines.... checked {cur_pair_num} pairs....")
                                print("current set length: ", len(pair_set), "    current memory use: ", cur_pair_set_memory, "MB")
                                print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    with open(args.output_filename, "w") as fh:
        for key in pair_set:
            fh.write(key + '\n')
        print("finished: in total ", len(pair_set), "pairs (dup filtered)")

    '''trial 1: extract the article list'''
    # art_dict = {}
    #
    # with open(args.input_filename, "r") as fh:
    #     line_num = 0
    #     cur_art_num = 0
    #     for line in fh:
    #         line_num += 1
    #         if line_num % 2 == 0:
    #             cur_arts = eval(line)
    #             for art in cur_arts:
    #                 lang = art[0].split("/")[-1][:2]
    #                 encode_pair_key = lang + "_"  + str(art[1])
    #                 art_dict[encode_pair_key] = 1
    #                 cur_art_num += 1
    #                 print(f"processing {line_num} lines.... checked {cur_art_num} articles (dup included)....")
    # with open(args.output_filename, "w") as fh:
    #     art_keys = art_dict.keys()
    #     for key in art_keys:
    #         fh.write(key + '\n')
    #     print("finished: in total ", len(art_keys), "artciles (dup filtered)")

    '''trial 2: deduplicate the pairs by once traverse'''
    # pair_dict = {}
    #
    # with open(args.input_filename, "r") as fh:
    #     line_num = 0
    #     cur_pair_num = 0
    #     for line in fh:
    #         line_num += 1
    #         if line_num % 2 == 0:
    #             cur_arts = eval(line)
    #             for art1 in cur_arts:
    #                 for art2 in cur_arts:
    #                     if art1 != art2:
    #                         lang1 = art1[0].split("/")[-1][:2]
    #                         lang2 = art2[0].split("/")[-1][:2]
    #                         # lang1_idx = lang_list.index(lang1)
    #                         # lang2_idx = lang_list.index(lang2)
    #                         # encode_lang_num = lang1_idx * 10 + lang2_idx
    #                         if art1[1] <= art2[1]:
    #                             encode_pair_key = lang1 + "_" + lang2 + "_" + str(art1[1]) + "_" + str(art2[1])
    #                         else:
    #                             encode_pair_key = lang2 + "_" + lang1 + "_" + str(art2[1]) + "_" + str(art1[1])
    #                         pair_dict[encode_pair_key] = 1
    #                         cur_pair_num += 1
    #                         print(f"processing {line_num} lines.... checked {cur_pair_num} pairs....")
    #
    # with open(args.output_filename, "w") as fh:
    #     pair_keys = pair_dict.keys()
    #     for key in pair_keys:
    #         fh.write(key + '\n')
    #     print("finished: in total ", len(pair_keys), "pairs")

