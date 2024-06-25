import datasets
import tqdm
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from argparse import ArgumentParser
from lemminflect import getAllInflections
# import spacy
# lemmatizer = WordNetLemmatizer()
# nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

def nltk_pos_tagger(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


def lemmatize_sentence(sentence, perword=False):
    sentence_words = nltk.word_tokenize(sentence)
    wordset = set()



ap = ArgumentParser(description='calc inc rate')
ap.add_argument('eval_file', type=str, help='References file -- multiple references separated ' +
                'by empty lines (or single-reference with no empty lines). Can also be a TSV ' +
                'file with source & reference columns. In that case, consecutive identical ' +
                'SRC columns are grouped as multiple references for the same source.')

ap.add_argument('--key', type=str, required=False, default="./common_gen-keys-dev.txt",
                help='References file -- multiple references separated ' +
                'by empty lines (or single-reference with no empty lines). Can also be a TSV ' +
                'file with source & reference columns. In that case, consecutive identical ' +
                'SRC columns are grouped as multiple references for the same source.')
args = ap.parse_args()

def get_all_inflections(key):
    inflection_sets = set(getAllInflections(key).values())
    inflections = set()
    inflections.add(key)
    for subset in inflection_sets:
        for item in subset:
            inflections.add(item)
    return inflections
N = 0
M = 0
with open(args.key, "r") as fkeys:
    with open(args.eval_file, "r") as ftest: #
        A, B = fkeys.readlines(), ftest.readlines()
        if len(A) == len(B):
            for (line_key, line_seq) in tqdm.tqdm(zip(A, B)):
                keys = [key for key in line_key.strip().split()]
                word_sets = set(nltk.word_tokenize(line_seq.strip().lower()))
                included_keys = []
                for key in keys:
                    key_inflections = get_all_inflections(key)
                    if len(set.intersection(key_inflections, word_sets)) > 0:
                        included_keys.append(key)
                    # else:
                    #     print(key_inflections)
                    #     print(word_sets)
                # print("{\"concept_set\": \"%s\", \"pred_scene\": [\"%s\"]}" % ("#".join(keys), line_seq.strip()), file=fout)
                # M += 1
                M += len(keys)
                if len(keys) == len(included_keys):
                    # N += 1
                    N += len(included_keys)
                else:
                    N += len(included_keys)
                    # print("Actual keys:", keys)
                    # print("Included keys:", included_keys)
                    # print("Generated samples:", line_seq.strip())
                    # print()
        else:
            i = 0
            j = 0
            for i in range(len(A)):
                cand_N = 0
                keys = A[i].strip().split()
                M += 1
                while B[j].strip() != "":
                    seq = lemmatize_sentence(B[j].strip().lower())
                    included_keys = [key for key in keys if seq.count(key) > 0]
                    cand_N = max(cand_N, len(included_keys))
                    j += 1
                if len(keys) == cand_N:
                    N += 1
                j += 1


print("stats:")
print(N)
print(M)
print(N / M)