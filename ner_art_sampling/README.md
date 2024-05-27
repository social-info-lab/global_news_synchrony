# Efficient pipeline for processing large-scale graph

## extract pairs from ne-art index with c++ code; the steps are: 
(1) filtering index; 
(2) build ne-art index; 
(3) extract the pairs; 
(4) filter pairs；
(5)embedding and similarity computation; 
(6) integrate duplicate pairs. 

**filter_index.py**(filter_index_script.sh)

This is to filtered out from the index the articles that have few than k wikified named entities, making sure the articles used in ne_art_index.py have a tf-idf score list for at least k wikified named entities.

**ne_art_index.py**(ne_art_index_script.sh)

This is to construct an index from each wikified named entity to the articles whose tf-idf score on this wikified named entity is among the top-k scores of all of each article's wikified named entities.

**extract_pair.cpp and cython_jaccard_sim.pyx**(c_extract_pair_script.sh)

This is to extract the pair from the ne_art_index. 

**pair_candidate.py**(pair_candidate_script.sh)

This is to filter the pairs that have low ne similarity or are duplicates from the extracted pairs.

**/../network_inference**

The "network_inference" folder includes the steps to embed news articles and compute their similarity.

**integrate_matched_inference.py**(integrate_matched_inference_script.sh)

This is to deduplicate the embedded pairs from the time spans that are overlapped for fitting the maximum memory of a running job when extracting the pairs.


**An extracted pair example:**
7722 18933391893345

The first 7 refers to the digit of the first article id;
The second 7 refers to the digit of the second article id;

The first 2 refers to the language of the first article mapping to the integer in lang_dict, i.e., {"es", 2} is referring to Spanish here;
The second 2 refers to the language of the second article mapping to the integer in lang_dict, i.e., {"es", 2} is referring to Spanish here;

So this example refers to an Spanish article with id "1893339" and an Spanish article with id "1893345".

--------------------------

# Faster NER sampling for both within a single language and across a pair of languages

## Instructions
If you meaningfully change a part of the code and you aim to generate new pairs of articles for annotation, then please:
* Update the description of the filtering/sampling methods in the `method` field in `refine_candidates.py`
* Github commit hash that marks the code that you will use is automatically stored at the time of execution in the `git_commit` field in `refine_candidates.py`


## Step-by-step

## Cython

We implemented our core operation (compute the jaccard similarity for two sorted int tuple) with Cython to speed the process up.

To run the scripts, firstly you have to setup the dynamic-link library: `python3 setup.py build_ext --inplace`

If you'd like to debug or analyze the cython code to further improve it, you can just run: `cython -a cython_jaccard_sim.pyx`


### create_index
`create_index.py` loops over all articles and creates a new output file that has one line per article in the format 
`out.write(f"{file}\t{lineno}\t{reladate}\t{url}\t{vec}\n")`

### creat_idf_dict
`creat_idf_dict.py` compute the inversed document frequency of words in the articles with repeat counting, and save them as a dictionary into json file

### create_offset_files
For each `.json` file create a `.offsets` file in the same directory.
A `.offsets` file has the same number of lines as the corresponding `.json` file. Each line is a single integer specifying the character offset of the line from the start of the file. This will enable us to easily jump to specific lines of any json file using the `seek` method of file objects.

### generate_candidates
multiprocess script to read the output of `create_index.py` and find candidate pairs of articles.
Each candidate article has a Jaccard similarity of at least 0.7 (completely arbitrary) based on the overlap of 2-letter NERs


### refine_candidates
Here we load in the full text of articles identified as candidates in `generate_candidates.py`. We use the `.offsets` files to jump to specific lines in files as needed.
This script will calculate the Jaccard similarity of the actually NERs (not the 2-letter approximations). 
It will also calculate Jaccard of the full text to exclude pairs that are near identical.

## Running the whole pipeline in practice

currently there is a little version inconsistence of index: for generating samples we have biased index files, which is `v3`. But when we are computing idf in refine_candidates, we need to use the index of the whole dataset. So it's still based on normal index, which is stored as `v2` on the server now. 

# for intra-lang sampling

Create indexes for each language (done already, no need to do it again unless the code changes):

`for lang in ar de es en fr it pl ru tr zh; do python3 create_index.py -i "/home/scott/ner/$lang/2020*.json" -o indexes/$lang & done`

Create idf dictionaries for each language (done already, no need to do it again unless the code changes):

`for lang in ar de es en fr it pl ru tr zh; do python3 create_idf_dict.py -o indexes/$lang & done`

Create offsets for each language (done already, no need to do it again unless the code changes):

`for lang in ar de es en fr it pl ru tr zh; do sudo python3 create_offset_files.py -i "/home/scott/ner/$lang/2020*.json"; done`

Generate 500 pairs in English (note that the full English index for 6 months can take up to 60% of memory)

`python3 generate_canidates.py -c1 -i indexes/en-full.index -n 500`

It's okay to run other languages in parallel, if you use 1 CPU for each language:

`for lang in ar de es fr pl ru tr zh; do python3 generate_canidates.py -c1 -i indexes/$lang-full.index -n 500 & done`

Refine all candidate pairs in English:

`for lang in ar de es en fr it pl ru tr zh; do python3 refine_candidates.py -i "candidates_$lang-full/*.jsonl" -l $lang -cl $lang & done`

See the URLs of the 10 refined pairs:

`head -n10 refined_en-full/n500_rank00.jsonl | jq -r '"URL1 " + .url1 + "\nURL2 " + .url2 + "\nSIM " + (.similarity|tostring) + "\n"' | less`

Exemplary results: `/home/ubuntu/mediacloud/ner_art_sampling`

# for inter-lang sampling

Create indexes for each language: 

`for lang in ar de es en fr it pl ru tr zh; `

`do python3 create_index.py -i "/home/scott/wikilinked/$lang/2020*.json.gz" -o indexes/$lang-wiki > log/create_index/$lang-wiki-$version_create_index_2021-09-14_16.log 2>&1 & done`

Create idf dictionaries for each language (done already, no need to do it again unless the code changes):

`for lang in ar de es en fr it pl ru tr zh; `

`do python3 create_idf_dict.py -o indexes/$lang-wiki & done`

Create offsets for each language:

`for lang in ar de es en fr it pl ru tr zh;` 

`do sudo python3 create_offset_files.py -i "/home/scott/wikilinked/$lang/2020*.json.gz" > log/create_offset/$lang-wiki-$version_create_offset_2021-09-14_16.log 2>&1 & done`

Generate candidates for a pair of language ($lang1 and $lang2):

`for lang1 and lang2 in ar de es en fr it pl ru tr zh; `

`do python3 generate_canidates.py -c1 -t inter-lang -i indexes/$lang1-wiki-$version1.index -cmp indexes/$lang2-wiki-$version2.index -n 500 > log/generate_canidates/candidates-v1_lang-$lang1-wiki-$version-$lang2-wiki-$version_2021-09-15_15.log 2>&1 & done`

Refine candidate pairs:

`for lang1 and lang2 in ar de es en fr it pl ru tr zh; `

`do python3 refine_candidates.py  -t inter-lang -i "candidates/candidates-$version_lang-$lang1-wiki-$version-$lang2-wiki-$version/*.jsonl" -l $lang1 -cl $lang2 > log/refine_candidates/refined-v1_candidates-v1_lang-$lang1-wiki-$version-$lang2-wiki-$version_2021-09-16_08.log 2>&1 & done`

# for bias dataset matching based sampling
example: directory names change from 'lang' to 'lang-bias', 'lang-wiki' to 'lang-wiki-bias'

create index:

`intra-lang:`
`python3 create_index.py -b -i "/home/scott/ner/pl/2020*.json" -o indexes/pl-bias > log/create_index/pl-bias-v3_create_index_2022-02-13_20.log 2>&1 &`

`inter-lang:`
`python3 create_index.py -b -i "/home/scott/wikilinked/pl/2020*.json.gz" -o indexes/pl-wiki-bias > log/create_index/pl-wiki-bias-v3_create_index_2022-02-13_20.log 2>&1 &`

create idf dict:
`just use the normal idf dict since the idf should be from the index of whole dataset instead of the bais dataset`

generate candidates: 

`intra-lang:`
`python3 generate_canidates.py -b -c1 -i indexes/de-wiki-bias-v3.index -n 10000 > log/generate_canidates/candidates-v7_lang-de-bias-v3_2022_02-18_17.log 2>&1 &`

`inter-lang:`
`python3 generate_canidates.py -b -c1 -t inter-lang -i indexes/de-wiki-bias-v3.index -cmp indexes/en-wiki-bias-v3.index -n 10000 > log/generate_canidates/candidates-v7_lang-de-wiki-bias-en-wiki-bias-v3_2022_02-18_17.log 2>&1 &`

refine candidates:
`python3 refine_candidates.py -b -i "candidates_$lang-full/*.jsonl" -l $lang -cl $lang`

## Version descrpition 
Create indexes: 
`earliest version: Not used index files. Directly match articles from json files`
`v1: index each article wtih file name, line number in the file, and the name entity list`

`v2: add temporal distance(days) to the first day into the index`

`v3: integrate the dataset with politics bias and country information (ABYZ/MBFC dataset) into create_index as an option`

`v4: add country information as a list and bias information of mbfc for each entry`

create idf dict:
`version aligns with creat_index`

Create offsets:
`No updates so far`

generate candidates: 
`earliest is from Xi Chen's oldest framework, which is not efficient enough in terms of time and memory`

`v1: updated pipeline with jaccard similarity of name entities`

`v2: add temporal window threshold`

`v3: add minimal jaccard threshold based on statistics`

`v4: correct the computation of name entity intersection`

`v5: compute jaccard similarity with repeat times of words`

`v6: go back to non-repeat jaccard similarity`

`v7: add bias index matching`

refine candidates:
`v1: basic filters e.g. url etc`

`v2: add minimal name entity number intersection as a filter`

`v3: add features e.g. tf-idf bm25 and corresponding logistic regression classifier`

## Reduced NE space (not used any more)

We're ultimately going to be looking at Jaccard similarity of named entities (NER). The space of NERs is too large, however, to sample efficiently.
Instead, we'll reduce the space by considering only vectors of 26^2=676. Each entry is the first and last letter of a NER. 

So, a document with the following NERs:

* Donald Trump
* United Kingdom
* John Smith

is reduced to

* dp
* um
* jh



## TODO
* apply translation ratio to all the languages
* Fix spelling of candidates in various places.
* Better distribution of work accross CPU cores in `generate_candidates`
* More memory efficient approach needed for `generate_candidates`. Perhaps use a FAISS index here.
* Consider whether there should be a minimum number of NERs in an article for it to be considered.

Done:

✔ Review completely arbitrary 0.7 threshold in `generate_candidates`\
✔ Push minimum lenth of articles (in words!) to `create_index` . Multiprocess this?\
✔ Discard duplicate URLs at `create_index` stage if possible.
