"""
Copied from https://github.com/neural-dialogue-metrics/Distinct-N/blob/master/distinct_n/metrics.py.
and be changed
"""
from metrics.n_grams import ngrams

__all__ = ["distinct_n_sentence_level", "distinct_n_corpus_level"]


def distinct_n_sentence_level(sentence, n):
    """
    Compute distinct-N for a single sentence.
    :param sentence: a list of words.
    :param n: int, ngram.
    :return: float, the metric value.
    """
    if len(sentence) == 0:
        return 0.0  # Prevent a zero division
    n_gram_seq = list(ngrams(sentence, n))
    distinct_ngrams = set(n_gram_seq)
    # print(n_gram_seq)

    return len(distinct_ngrams) / len(n_gram_seq)


def distinct_n_corpus_level(sentences, n):
    """
    Compute average distinct-N of a list of sentences (the corpus).
    :param sentences: a list of sentence.
    :param n: int, ngram.
    :return: float, the average value.
    """
    splited_sentences = [s.split() for s in sentences]
    return sum(distinct_n_sentence_level(sentence, n) for sentence in splited_sentences) / len(sentences)


def dist_compute(sentences, n_list):
    result_dict = {}
    for n in n_list:
        result_dict["dist-%d" % n] = distinct_n_corpus_level(sentences, n)
    return result_dict

