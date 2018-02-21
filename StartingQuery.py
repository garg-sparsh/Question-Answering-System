from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize,word_tokenize
from string import punctuation
from heapq import nlargest
import pickle
import functools
import math
from collections import defaultdict
import sys
questionwords = ["who", "what", "where", "when", "why", "how", "whose", "which", "whom"]
yesnowords = ["can", "could", "would", "is", "does", "has", "was", "were", "had", "have", "did", "are", "will", "list", "which"]
commonwords = ["the", "a", "an", "is", "are", "were", "."]
characters = " .,!#$%^&*();:\n\t\\\"?!{}[]<>"
global ques
ques = {}
global ans
ans = {}
global ideal_ans
ideal_ans = {}
with open('factoid_ques', 'rb') as input_que:
    ques = pickle.load(input_que)
with open('factoid_ans', 'rb') as input_ans:
    ans = pickle.load(input_ans)
with open('factoid_ideal.txt', 'rb') as input_ans:
    ideal_ans = pickle.load(input_ans)
with open('articles', 'rb') as input_que:
    articles = pickle.load(input_que)
N = len(ques)
dictionary_fact = set()
postings_fact = defaultdict(dict)
dictionary_complex = set()
postings_complex = defaultdict(dict)
document_frequency = defaultdict(int)
document_frequency_complex = defaultdict(int)
length = defaultdict(float)
length_complex = defaultdict(float)
length_ideal = defaultdict(float)
dictionary_ideal = set()
postings_ideal = defaultdict(dict)
document_frequency_ideal = defaultdict(int)

class FrequencySummarizer:
  def __init__(self, min_cut=0.1, max_cut=0.9):

    self._min_cut = min_cut
    self._max_cut = max_cut
    self._stopwords = set(stopwords.words('english') + list(punctuation))

  def _compute_frequencies(self, word_sent):

    freq = defaultdict(int)
    for s in word_sent:
      for word in s:
        if word not in self._stopwords:
          freq[word] += 1
    m = float(max(freq.values()))
    for w in list(freq.keys()):
        freq[w] = freq[w]/m
        if freq[w] >= self._max_cut or freq[w] <= self._min_cut:
            del freq[w]
    return freq

  def summarize(self, text, n):

    sents = sent_tokenize(text)
    assert n <= len(sents)
    word_sent = [word_tokenize(s.lower()) for s in sents]
    self._freq = self._compute_frequencies(word_sent)
    ranking = defaultdict(int)
    for i,sent in enumerate(word_sent):
      for w in sent:
        if w in self._freq:
          ranking[i] += self._freq[w]
    sents_idx = self._rank(ranking, n)
    return [sents[j] for j in sents_idx]

  def _rank(self, ranking, n):
    return nlargest(n, ranking, key=ranking.get)

def tokenize(document):
    terms = document.lower().split()
    return [term.strip(characters) for term in terms]

def processquestion(qwords):
    # Find "question word" (what, who, where, etc.)
    questionword = ""
    qidx = -1

    for (idx, word) in enumerate(qwords):
        if word.lower() in questionwords:
            questionword = word.lower()
            qidx = idx
            break
        #elif word.lower() in yesnowords:
           # return ("YESNO", qwords)

    if qidx < 0:
        return ("MISC", qwords)

    if qidx > len(qwords) - 3:
        target = qwords[:qidx]
    else:
        target = qwords[qidx + 1:]
    type = "MISC"
    # Determine question type
    if questionword in ["who", "whose", "whom"]:
        type = "PERSON"
    elif questionword == 'where':
        type = "PLACE"
    elif questionword == "when":
        type = "TIME"
    elif questionword == "how":
        if target[0] in ["few", "little", "much", "many"]:
            type = "QUANTITY"
            target = target[1:]
        elif target[0] in ["young", "old", "long"]:
            type = "TIME"
            target = target[1:]

    # Trim possible extra helper verb
    if questionword == "which":
        target = target[1:]
    if target[0] in yesnowords:
        target = target[1:]

    # Return question data
    target = [word for word in target if not word in stopwords.words('english')]
    return (type, target)

def union(sets,type):
    """Returns the intersection of all sets in the list sets. Requires
    that the list sets contains at least one element, otherwise it
    raises an error."""
    if type == 'fact':
        return functools.reduce(set.union, [s for s in sets])
    else:
        return functools.reduce(set.union, [s for s in sets])

def vector_space():
    global dictionary_fact, postings_fact, dictionary_complex, postings_complex, dictionary_ideal, postings_ideal
    for id in ques.keys():
        terms = tokenize(ques[id])
        unique_terms = set(terms)
        dictionary_fact = dictionary_fact.union(unique_terms)
        for term in unique_terms:
            postings_fact[term][id] = terms.count(term)
    for id in articles.keys():
        terms = tokenize(articles[id])
        unique_terms = set(terms)
        dictionary_complex = dictionary_complex.union(unique_terms)
        for term in unique_terms:
            postings_complex[term][id] = terms.count(term)
    for id in ideal_ans.keys():
        terms = tokenize(str(ideal_ans[id]).replace('[','').replace(']','').replace(',',''))
        unique_terms = set(terms)
        dictionary_ideal = dictionary_ideal.union(unique_terms)
        for term in unique_terms:
            postings_ideal[term][id] = terms.count(term)

def tf():
    global document_frequency, document_frequency_complex
    for term in dictionary_fact:
        document_frequency[term] = len(postings_fact[term])
    for term in dictionary_complex:
        document_frequency_complex[term] = len(postings_complex[term])
    for term in dictionary_ideal:
        document_frequency_ideal[term] = len(postings_ideal[term])

def idf(term,document):
    if document == 'fact':
        if term in dictionary_fact:
            return math.log(N/document_frequency[term],2)
        else:
            return 0.0
    elif document == 'complex':
        if term in dictionary_complex:
            return math.log(N / document_frequency_complex[term], 2)
        else:
            return 0.0
    else:
        if term in dictionary_ideal:
            return math.log(N / document_frequency_ideal[term], 2)
        else:
            return 0.0

def importance(term,id,type):
    if type == 'fact':
        if id in postings_fact[term]:
            return postings_fact[term][id]*idf(term,'fact')
        else:
            return 0.0
    elif type == 'complex':
        if id in postings_complex[term]:
            return postings_complex[term][id] * idf(term, 'complex')
        else:
            return 0.0
    else:
        if id in postings_ideal[term]:
            return postings_ideal[term][id] * idf(term, 'ideal')
        else:
            return 0.0

def initialize_lengths():
    global length, length_complex
    for id in ques.keys():
        l = 0
        for term in dictionary_fact:
            l += importance(term,id,'fact')**2
        length[id] = math.sqrt(l)
    for id in articles.keys():
        l = 0
        for term in dictionary_complex:
            l += importance(term,id,'complex')**2
        length_complex[id] = math.sqrt(l)
    for id in ideal_ans.keys():
        l = 0
        for term in dictionary_ideal:
            l += importance(term,id,'ideal')**2
        length_ideal[id] = math.sqrt(l)

def cosine_similarity(query,id):
    similarity = 0.0
    for term in query:
        if term in dictionary_fact:
            similarity += idf(term,'fact')*importance(term,id,'fact')
    similarity = similarity / length[id]
    return similarity

def cosine_similarity_complex(query,id):
    similarity = 0.0
    for term in query:
        if term in dictionary_complex:
            similarity += idf(term,'complex')*importance(term,id,'complex')
    similarity = similarity / length_complex[id]
    return similarity

def cosine_similarity_ideal(query,id):
    similarity = 0.0
    for term in query:
        if term in dictionary_ideal:
            similarity += idf(term,'ideal')*importance(term,id,'ideal')
    similarity = similarity / length_ideal[id]
    return similarity

def answercomplex(query):
    relevant_document_ids_complex = union(
        [set(postings_complex[term].keys()) for term in query], 'complex')
    if not relevant_document_ids_complex:
        return ""
    else:
        scores = sorted([(id, cosine_similarity_complex(query, id))
                         for id in relevant_document_ids_complex],
                        key=lambda x: x[1],
                        reverse=True)
        (id, score) = scores[0]
        return id, score

def do_search():
    print()
    query = tokenize(input("Your question:"))
    print()
    que_type = query[0]
    (type, target) = processquestion(query)
    query = target
    if query == []:
        sys.exit()
    relevant_document_ids = union(
            [set(postings_fact[term].keys()) for term in query],'fact')
    relevant_ideal_ans_ids = union(
            [set(postings_ideal[term].keys()) for term in query],'fact')
    if not relevant_document_ids and not relevant_ideal_ans_ids:
        try:
            id_complex, score_complex = answercomplex(query)
            return articles[id_complex]
        except:
            return "This is not upto my knowledge"
    else:
        scores = sorted([(id,cosine_similarity(query,id))
                         for id in relevant_document_ids],
                        key=lambda x: x[1],
                        reverse=True)
        scores_ideal = sorted([(id,cosine_similarity_ideal(query,id))
                         for id in relevant_ideal_ans_ids],
                        key=lambda x: x[1],
                        reverse=True)
        if scores !=[]:
            (id, score) = scores[0]
        else:
            score = 0.1
            id = None
        if scores_ideal != []:
            id_ideal, score_ideal = scores_ideal[0]
        else:
            score_ideal = 0.1
            id_ideal = None
        if score > 10:
            if que_type.lower() in yesnowords:
                return ((str(ans[id]).replace('[','').replace(']','').replace("'",'')))
            else:
                return ((str(ideal_ans[id]).replace('[', '').replace(']', '').replace("'", '')))
        else:
            try:
                id_complex, score_complex = answercomplex(query)
                if score_complex > score_ideal:
                    ps = FrequencySummarizer()
                    if articles[id_complex] == "" or articles[id_complex] is None:
                        return "This is not upto my knowledge"
                    a = ps.summarize(articles[id_complex], 2)
                    a = str(a).replace('[', '').replace(']', '').replace("'", '')
                    return a
                else:
                    if id_ideal != None:
                        return ((str(ideal_ans[id_ideal]).replace('[', '').replace(']', '').replace("'", '')))
            except:
                print('Not able to understand the question')
                pass

def main():
    vector_space()
    try:
        tf()
        initialize_lengths()
        while True:
            answer = do_search()
            if answer != None:
                print("Bot:",answer)
    except:
        print()
        print('Shutting down')

if __name__=='__main__':
    main()
