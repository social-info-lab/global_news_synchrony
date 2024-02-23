from argparse import ArgumentParser
import os

# save the entity-art index to files as per each article,
# and then try to deduplicate pairs accord to:
# (1) certain line number inequality,
# e.g. en_1000 and es_2000 will only show in en_1000 file but not in es_2000 file
# (2) list to set to filter dupllicates.

lang_list = ['en','de','es','pl','zh','fr','ar','tr','it','ru']

if __name__ == '__main__':
    a = [1,2,3]
    b = str(a)

    parser = ArgumentParser()
    parser.add_argument("-i", "--input-filename", dest="input_filename",
                        default="indexes/update-ne-art-wiki.index", type=str,
                        help="Index filename.")
    args = parser.parse_args()

    output_folder = "big_data_art/"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # extract the article list
    art_dict = {}

    with open(args.input_filename, "r") as fh:
        line_num = 0
        cur_art_num = 0
        for line in fh:
            line_num += 1
            if line_num % 2 == 0:
                cur_arts = eval(line)
                for art in cur_arts:
                    lang = art[0].split("/")[-1][:2]
                    encode_pair_key = lang + "_"  + str(art[1])
                    with open(output_folder + encode_pair_key +".txt", "a+") as fout:
                        fout.writelines(line)
                        fout.write('\n')

                    cur_art_num += 1
                    print(f"processing {line_num} lines.... checked {cur_art_num} articles (dup included)....")