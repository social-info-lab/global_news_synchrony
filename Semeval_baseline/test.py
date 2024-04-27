import collections
import csv

if __name__ == "__main__":
    input_directory = "semeval_8_2021_ia_data/train"
    output_directory = "semeval_8_2021_ia_data/ner/train"
    csv_file = "semeval-2022_task8_train-data_batch.csv"

    # loading language from the csv file
    lang_url_dict = collections.defaultdict(list)
    url_lang_dict = collections.defaultdict(str)
    # csv_file = "semeval-2022_task8_eval_data_202201.csv"
    csv_reader = csv.reader(open(csv_file))
    for line in csv_reader:
        lang_url_dict[line[0]].append(line[3])
        lang_url_dict[line[0]].append(line[5])
        lang_url_dict[line[1]].append(line[4])
        lang_url_dict[line[1]].append(line[6])

        url_lang_dict[line[3]] = line[0]
        url_lang_dict[line[5]] = line[0]
        url_lang_dict[line[4]] = line[1]
        url_lang_dict[line[6]] = line[1]
    print()