from sentence_transformers import SentenceTransformer
import pandas as pd
import torch
from torch import Tensor
import spacy

from nltk.data import load


model = SentenceTransformer('sentence-transformers/multi-qa-MiniLM-L6-cos-v1')
sbert_model = SentenceTransformer('sentence-transformers/multi-qa-MiniLM-L6-cos-v1') 

def extract_title_from_context(context_title):
    context_title = context_title.strip()
    if not context_title.startswith('('):
        return "", context_title
    queue_flag = 0
    for i in range(len(context_title) - 1):
        if context_title[i] == '(':
            queue_flag += 1
        elif context_title[i] == ')':
            queue_flag -= 1
            if queue_flag == 0:
                c_ = context_title[i + 1:].strip()
                title = context_title[1:i]
                return title, c_
                break
    return "", context_title

def word_embedding(my_str):
    w_embedding = model.encode(my_str)
    return w_embedding

def is_simple_sentence(sent):
    if ", " in sent or "; " in sent: 
        return False
    return True

def before_preprocess(sent):
    sent = sent.replace(", however, ", " ")
    sent = sent.replace(", but, ", " ")
    conjunctions_list = ["however", "but", "otherwise", "whereas", "besides", "in addition", "even so",
                         "thus", "hence", "therefore", "furthermore", "moreover", "what's more", "anyway", "usually",
                         "also", "additionally", "specifically", "nevertheless", "firstly", "secondly", "thirdly"]
    for conj in conjunctions_list:
        if sent.lower().startswith(conj + ","):
            sent = sent[len(conj) + 1:].strip()
    return sent

# Standard sentence tokenizer.
def sent_tokenize(text, language="english"):
    """
    Return a sentence-tokenized copy of *text*,
    using NLTK's recommended sentence tokenizer
    (currently :class:`.PunktSentenceTokenizer`
    for the specified language).

    :param text: text to split into sentences
    :param language: the model name in the Punkt corpus
    """
    tokenizer = load(f"tokenizers/punkt/{language}.pickle")
    return tokenizer.tokenize(text)



def get_sentence_from_a_context(context):
    res_sent_list = []
    context = context.strip()
    context = context.replace("(i.e. ", "(")  # this rule to avoid wrong sentence split by sent_tokenize().
    sent_list = sent_tokenize(context)
    for s in sent_list:
        s = s.strip()
        s = before_preprocess(s)
        if s.strip().endswith("?"):  # remove question in context.
            continue
        if not is_simple_sentence(s):  # skip complex sentence.
            continue
        if s.endswith("; "):
            s = s[:-2] + "."
        res_sent_list.append(s)
    return res_sent_list

def dot_score(a: Tensor, b: Tensor):
    """
    Computes the dot-product dot_prod(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = dot_prod(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    return torch.mm(a, b.transpose(0, 1))

def calculate_sim_origin_sentence(origin, target_list, context_embedding, top_n=1):
    '''
    This method can calculate the similarity between origin and target.
    :param origin:  a word or a sentence.
    :param target_list:  a list containing words or sentences.
    :param top_n:  the number of return list.
    :param context_embedding: the embedding of target_list.
    :return: 2-dim list  --- the top_n most similar target words/sentences for origin; e.g., [('dog', 0.9), ('cat', 0.85)]
    '''
    w_embedding = sbert_model.encode(origin)

    sim = dot_score(w_embedding, context_embedding).tolist()[0]
    
    max_sim = -2
    max_sim_index = 0
    for i, score in enumerate(sim):
        if score > max_sim:
            max_sim = score
            max_sim_index = i
    
    return target_list[max_sim_index], max_sim


def get_sentences_from_file(file_path, all_sentences = []):
    data = pd.read_csv(file_path, sep='\t', header=None, names=['q_c', 'a'], on_bad_lines='skip')
    c = 0
    for ind in data.index:
        context_title = data['q_c'][ind].split("\\n")[1]
        title, context = extract_title_from_context(context_title)
        sentences_in_context = get_sentence_from_a_context(context)
        all_sentences.extend(sentences_in_context)
    return all_sentences

def get_all_sentences_context_dataset():
    l1 = get_sentences_from_file('contxt_data/data_boolq_dev.tsv')
    print("Read file data_boolq_dev.tsv, sentences length", len(l1))
    l2 = get_sentences_from_file('contxt_data/data_boolq_train.tsv', l1)
    print("Read file data_boolq_train.tsv, sentences length", len(l2))
    l3 = get_sentences_from_file('contxt_data/data_narrative_dev.tsv', l2)
    print("Read file data_narrative_dev.tsv, sentences length", len(l3))
    l4 = get_sentences_from_file('contxt_data/data_narrative_train.tsv', l3)
    print("Read file data_narrative_train.tsv, sentences length", len(l4))
    l5 = get_sentences_from_file('contxt_data/data_squad2_dev.tsv', l4)
    print("Read file data_squad2_dev.tsv, sentences length", len(l5))
    l6 = get_sentences_from_file('contxt_data/data_squad2_train.tsv', l5)
    print("Read file data_squad2_train.tsv, sentences length", len(l6))
    return l6
