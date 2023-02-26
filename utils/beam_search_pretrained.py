# From https://github.com/JunjieHu/ReCo-RL and changed

import os
import sys
dir_name = os.path.abspath(os.path.join(os.path.dirname("__name__"), "../"))
sys.path.append(dir_name)
from config.CONSTANTS import *
from utils.beam_search_utils import repeat, topK_2d_ngrams, update_ngrams, select_hid, topK_2d, new_state,\
    update_complete_hid, select_sequences_by_pointer, update_top_seqs, select_hid2, assert_pre_end_token, repeat_tuple,\
    repeat_recur, get_tensor_from_index
import torch
from collections import defaultdict


def pretrained_beam_search(model, context_tuple, beam_size=5, decode_min_len=30, decode_max_len=150, force_eos=100,
                avoid_ngram=2, to_word=True,
                pad_index=PAD_INDEX, end_index=END_INDEX, unk_index=UNK_INDEX, start_index=START_INDEX,
                tensor_index_list=None):
    if tensor_index_list is None:
        device = context_tuple[0].device
        batch_size = context_tuple[0].size(0)
        new_tensor = context_tuple[0].new_empty
    else:
        use_src_tensor = get_tensor_from_index(context_tuple, tensor_index_list)
        assert type(use_src_tensor) == torch.Tensor
        device = use_src_tensor.device
        batch_size = use_src_tensor.size(0)
        new_tensor = use_src_tensor.new_empty

    num_vocab = model.decoder_vocab_size

    pre_end_ids = model.decoder_tokenizer.encode(PRE_END_TOKEN_LIST, add_special_tokens=False)
    not_word_token_ids = [pad_index, end_index]

    repeated_context = repeat_recur(context_tuple, dim=1, k=beam_size)  # (batch_size * beam_size, length,  hidden size)
    dec_input = [[]] * batch_size * beam_size

    top_score = new_tensor(batch_size, beam_size).fill_(-float('inf'))  # [batch, beam]
    top_score.data[:, 0].fill_(0)

    pad_vec = new_tensor(1, num_vocab).float().fill_(-float('inf'))
    pad_vec[0, pad_index] = 0
    eos_vec = new_tensor(1, num_vocab).float().fill_(-float('inf'))
    eos_vec[0, end_index] = 0
    end_mark = new_tensor(batch_size, beam_size).byte().fill_(0)

    eos_mask = new_tensor(batch_size, beam_size).byte().fill_(0)  # [batch, beam_size]
    sent_len = new_tensor(batch_size, beam_size).long().fill_(0)

    top_seqs = [[[] for _ in range(beam_size)] for _ in range(batch_size)]  # [batch, beam, num_words]

    ngrams = [[defaultdict(list) for _ in range(beam_size)] for _ in range(batch_size)]

    for i in range(decode_max_len):
        log_prob = model.single_decode(repeated_context, dec_input)
        log_prob = log_prob.view(batch_size, beam_size, -1)  # (batch_size, beam_size, vocab_size)
        log_prob.data[:, :, unk_index].fill_(-float('inf'))
        if eos_mask.sum() > 0:
            log_prob.data.masked_scatter_(eos_mask.unsqueeze(2), pad_vec.expand((eos_mask.sum(), num_vocab)))

        if i > decode_min_len and end_mark.sum() > 0:
            log_prob.data.masked_scatter_(end_mark.unsqueeze(2), eos_vec.expand((end_mark.sum(), num_vocab)))

        if avoid_ngram > 0:
            nm1_grams = []
            log_mask = torch.zeros((batch_size, beam_size, num_vocab), dtype=torch.uint8, device=device)
            for ii, beam_seq in enumerate(top_seqs):  # batch
                for jj, seq in enumerate(beam_seq):  # beam
                    nm1_gram = tuple(seq[-(avoid_ngram - 1):])
                    exist_words = ngrams[ii][jj][nm1_gram]  # a list of existing n-th word
                    for widx in exist_words:
                        log_mask[ii][jj][widx] = 1

            log_prob.data.masked_fill_(log_mask.eq(1), -float('inf'))

        if i < decode_min_len:
            log_prob[:, :, end_index].fill_(-float("inf"))

        if i == 0:
            score = top_score.unsqueeze(2) + log_prob
        else:
            score = ((top_score * sent_len).unsqueeze(2) + log_prob) / (sent_len + 1 - eos_mask.float()).unsqueeze(2)

        top_score, top_rowid, top_colid = topK_2d(score)  # [batch, beam_size]

        eos_mask = eos_mask.gather(dim=1, index=top_rowid.data) | top_colid.data.eq(end_index)

        top_seqs = update_top_seqs(top_seqs, top_rowid.tolist(), top_colid.tolist())  # [batch, beam, num_words]
        dec_input = torch.tensor(top_seqs, dtype=torch.long)
        dec_input = dec_input.view(batch_size * beam_size, -1)
        dec_input = dec_input.tolist()

        new_ngrams = []
        for ii, (bi_bptr, bi_ngram) in enumerate(zip(top_rowid.tolist(), ngrams)):
            new_bi_ngrams = []
            for jj, bptr in enumerate(bi_bptr):
                ngram_copy = defaultdict(list, {k: [w for w in v] for k, v in bi_ngram[bptr].items()})
                new_bi_ngrams.append(ngram_copy)
            new_ngrams.append(new_bi_ngrams)
        ngrams = new_ngrams

        if avoid_ngram > 0:
            for ii, bi_seqs in enumerate(top_seqs):
                for jj, seqs in enumerate(bi_seqs):
                    if seqs[-1] == pad_index or seqs[-1] == end_index:
                        continue
                    nm1_ngram = tuple(seqs[-avoid_ngram:-1])
                    ngrams[ii][jj][nm1_ngram].append(seqs[-1])

        sent_len = sent_len.gather(dim=1, index=top_rowid.data) + (1 - eos_mask.long())

        end_mark = (sent_len > force_eos) & assert_pre_end_token(top_colid, pre_end_ids) & (~eos_mask.eq(1))

        if eos_mask.sum() == batch_size * beam_size:
            break

    sequence = top_seqs
    sort_score = top_score
    if to_word:
        sequence = [[model.decoder_tokenizer.convert_ids_to_tokens(seq,  skip_special_tokens=True) for seq in beam]
                    for beam in sequence]
        sort_score = sort_score.cpu().data.numpy().tolist()

    return sequence, sort_score

