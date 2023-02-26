import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict


def get_tensor_from_index(context_tuple, tensor_index_list):
    tmp = context_tuple
    for i in tensor_index_list:
        tmp = tmp[i]
    return tmp


def new_state(hidden_state):
    if type(hidden_state) is tuple or type(hidden_state) is list:
        new_state = []
        for h in hidden_state:
            nh = h.data.new(h.size()).fill_(0)
            new_state.append(nh)
        new_state = tuple(new_state)
    else:
        new_state = hidden_state.data.new(hidden_state.size()).fill_(0)
    return new_state


def repeat(input, dim, k):
    """Repeat the input tensor k times along the dim dimention
        input: [dim, d]
        output: [dim*k, d]
    """
    if type(input) is tuple:
        size = [-1] * len(input[0].size())
        size.insert(dim, k)
        new_size = list(input[0].size())
        new_size[dim - 1] *= k
        input = list(input)
        for idx in range(len(input)):
            input[idx] = input[idx].unsqueeze(dim).expand(tuple(size)).contiguous().view(new_size)
        input = tuple(input)
    else:
        size = [-1] * len(input.size())
        size.insert(dim, k)
        new_size = list(input.size())
        new_size[dim - 1] *= k
        input = input.unsqueeze(dim).expand(tuple(size)).contiguous().view(new_size)
    return input


def repeat_tuple(input_data, dim, k):
    assert type(input_data) is tuple
    result_list = []
    for s_i in input_data:
        result_list.append(repeat(s_i, dim, k))
    return tuple(result_list)


def repeat_recur(input_data, dim, k):
    if type(input_data) in (list, tuple):
        result_list = []
        for x in input_data:
            result_list.append(repeat_recur(x, dim, k))
        return result_list
    elif type(input_data) == int:
        return input_data
    elif type(input_data) == torch.Tensor:
        return repeat(input_data, dim, k)
    else:
        raise ValueError


def resize(input, size):
    if type(input) is tuple:
        input = list(input)
        for idx in range(len(input)):
            input[idx] = input[idx].view(size)
        input = tuple(input)
    else:
        input.view(size)
    return input


def topK(score):
    """
    Args:
        score: [beam, num_vocab]
    Return:
        top_score: [beam]
        top_rowid: [beam], beam id
        top_colid: [beam], word id
    """
    beam_size, num_vocab = score.size()
    flat_score = score.view(beam_size * num_vocab)
    top_score, top_index = flat_score.topk(k=beam_size, dim=0)
    top_rowid, top_colid = top_index // num_vocab, top_index % num_vocab
    return top_score, top_rowid, top_colid


def topK_2d(score):
    """
    Args:
        score: [batch, beam_size, num_vocab]
    Return:
        top_score: [batch, beam_size], select score
        top_rowid: [batch, beam_size], beam id
        top_colid: [batch, beam_size], word id
    """
    batch_size, beam_size, num_vocab = score.size()
    flat_score = score.view(batch_size, beam_size * num_vocab)
    top_score, top_index = flat_score.topk(k=beam_size, dim=1)
    top_rowid, top_colid = top_index // num_vocab, top_index % num_vocab
    return top_score, top_rowid, top_colid


def copy_dict(d):
    di = defaultdict(list)
    for k, v in d.items():
        di[k] = [vi for vi in v]
    return di


def topK_2d_ngrams(orig_score, ngrams, sequence, n=2, word_seq=None, word_ngrams=None, vocab=None):
    """
    Args:
        orig_score: [batch, beam_size, num_vocab]
        ngrams: [batch, beam_size], each is a set
        sequence: [batch, beam_size]
    Return:
        top_score: [batch, beam_size]
        top_rowid: [batch, beam_size]
        top_colid: [batch, beam_size]
    """
    batch_size, beam_size, num_vocab = orig_score.size()
    flat_score = orig_score.view(batch_size, beam_size * num_vocab)
    top_score, top_index = torch.sort(flat_score, dim=1, descending=True)  # [batch, beam*|V|]

    if ngrams is None:
        new_score = top_score[:, 0:beam_size]
        new_index = top_index[:, 0:beam_size]
        new_rowid, new_colid = new_index // num_vocab, new_index % num_vocab
        for i in range(batch_size):
            for j in range(beam_size):
                sequence[i][j] = sequence[i][j] + [int(new_colid[i][j])]
    else:
        new_tensor = orig_score.data.new
        new_score = new_tensor(batch_size, beam_size).zero_()
        new_rowid = new_tensor(batch_size, beam_size).zero_().long()
        new_colid = new_tensor(batch_size, beam_size).zero_().long()

        for i in range(batch_size):
            b = 0
            seq_b = sequence[i][b]
            ngram_b = ngrams[i][b]
            wseq_b = word_seq[i][b]
            wngram_b = word_ngrams[i][b]
            for j in range(top_score.size(1)):
                score = top_score[i, j].item()
                index = top_index[i, j].item()
                rowidx = int(index / num_vocab)
                colidx = int(index % num_vocab)
                pre_word = seq_b[-(n - 1):]
                cur_word = colidx
                ngram = tuple(pre_word + [cur_word])
                pre_w = wseq_b[-(n - 1):]
                cur_w = vocab.id2word[colidx]
                wngram = tuple(pre_w + [cur_w])
                tmp = wngram in wngram_b
                if ngram in ngram_b:
                    continue
                ngram_b.add(ngram)
                wngram_b.add(wngram)
                new_score[i][b] = score
                new_rowid[i][b] = rowidx
                new_colid[i][b] = colidx
                sequence[i][b] = sequence[i][b] + [colidx]
                word_seq[i][b] = word_seq[i][b] + [cur_w]
                b += 1
                if b >= beam_size:
                    break
                seq_b = sequence[i][b]
                ngram_b = ngrams[i][b]
                wseq_b = word_seq[i][b]
                wngram_b = word_ngrams[i][b]
    return new_score, new_rowid, new_colid, ngrams, sequence


def update_ngrams(top_rowid, top_colid, top_seqs, ngrams, n):
    batch_size, beam_size = len(top_seqs), len(top_seqs[0])
    new_seqs = [[[] for _ in range(beam_size)] for _ in range(batch_size)]
    new_ngrams = [[set() for _ in range(beam_size)] for _ in range(batch_size)]
    for i in range(batch_size):
        for j in range(beam_size):
            beam = int(top_rowid[i][j])
            word = int(top_colid[i][j])
            new_seqs[i][j] = top_seqs[i][beam] + [word]
            new_ngrams[i][j] = ngrams[i][beam].copy()
            new_ngrams[i][j].add(top_seqs[i][beam][-n:])
    return new_seqs, new_ngrams


def select_hid_1d(hidden, row_id):
    """
    Args:
        hidden: [beam, hidden_size]
        row_id: [beam]
    Return:
        new_hidden: [beam, hidden_size]
    """
    new_hidden = []
    for h in hidden:
        new_h = h[row_id.data]
        new_hidden.append(new_h)
    new_hidden = tuple(new_hidden)
    return new_hidden


def select_hid(hidden, batch_id, row_id):
    """ Re-arange the hidden state according to the selected beams in the previous step
    Args:
        hidden: [batch*beam_size, hidden_size]
        batch_id: [batch, beam_size]
        row_id: [batch, beam_size]
    Return:
        new_hidden: [batch*beam_size, hidden_size]
    """
    batch_size, beam_size = row_id.size()
    if type(hidden) is tuple or type(hidden) is list:
        new_hidden = []
        for h in hidden:
            new_h = h.view(batch_size, beam_size, -1)[batch_id.data, row_id.data]
            new_h = new_h.view(batch_size * beam_size, -1)
            new_hidden.append(new_h)
        new_hidden = tuple(new_hidden)
    else:
        new_hidden = hidden.view(batch_size, beam_size, hidden.size(2))[:, batch_id.data, row_id.data]
        new_hidden = new_hidden.view(batch_size * beam_size, hidden.size(2))
    return new_hidden


def select_hid2(hidden, batch_id, row_id):
    batch_size, beam_size = row_id.size()
    new_h = hidden.view(batch_size, beam_size, -1)[batch_id.data, row_id.data]
    new_h = new_h.view(batch_size * beam_size, -1)
    return new_h


def update_complete_hid(last_complete_hid, eos_mask, cur_dec_hid):
    """
    Args:
        last_complete_hid: [batch*beam, hidden_size]
        eos_mask: [batch, beam]
        cur_dec_hid: [batch*beam, hidden_size]
    Return:
        complete_hid: [batch*beam, hidden_size]
    """
    hidden_size = last_complete_hid[0].size(-1)
    eos = eos_mask.view(-1, 1).repeat(1, hidden_size)
    if type(last_complete_hid) is tuple or type(last_complete_hid) is list:
        complete_hid = []
        for lch, cdh in zip(last_complete_hid, cur_dec_hid):
            h = eos.float() * lch + (1 - eos).float() * cdh
            complete_hid.append(h)
        complete_hid = tuple(complete_hid)
    else:
        complete_hid = eos.float() * last_complete_hid + (1 - eos).float() * cur_dec_hid
    return complete_hid


def update_top_seqs(top_seqs, beam_bptr, top_words):
    """
    Args:
        top_seqs: [batch, beam, num_words]
        beam_bptr: [batch, beam]
        top_words: [batch, beam]
    Return:
        top_seqs: [batch, beam, num_words+1]
    """
    new_seqs = []
    for bi_seqs, bi_bptr, bi_words in zip(top_seqs, beam_bptr, top_words):
        new_bi_seqs = [bi_seqs[bptr] + [word] for bptr, word in zip(bi_bptr, bi_words)]
        new_seqs.append(new_bi_seqs)
    return new_seqs


def select_sequences_by_pointer(completed_sequences, completed_scores, sequences, scores, beam_bptr):
    """
    Args:
        completed_sequences: [batch, beam, num_seq, max_seq_len]
        completed_scores: [batch, beam, num_seq]
        sequences: [batch, beam, max_seq_len］
        scores: [batch, beam］
        beam_bptr: [batch, beam]
    """
    if len(completed_sequences) == 0:
        completed_sequences = [[[seq] for seq in beams] for beams in sequences]
        completed_scores = [[[score] for score in beams] for beams in scores]
        return completed_sequences, completed_scores
    else:
        new_sequences, new_scores = [], []
        for cseqs, csc, seqs, sc, bptr in zip(completed_sequences, completed_scores, sequences, scores, beam_bptr):
            ncseqs = [cseqs[b] + [s] for b, s in zip(bptr, seqs)]
            ncsc = [csc[b] + [s] for b, s in zip(bptr, sc)]
            new_sequences.append(ncseqs)
            new_scores.append(ncsc)
    return new_sequences, new_scores


def assert_pre_end_token(top_colid, pre_token_ids):
    start = top_colid.new_zeros(top_colid.shape, dtype=torch.bool).fill_(False)
    for i in pre_token_ids:
        start = start | top_colid.eq(i)
    return start



