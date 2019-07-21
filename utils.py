"""

Some utilities for reading files and formatting sequences

"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import functools
from itertools import islice, chain

import numpy as np
import Bio.SeqIO

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical


IUPAC_CODES = list('ACDEFGHIKLMNPQRSTVWY*')
input_symbols = {label: i for i, label in enumerate(IUPAC_CODES)}
def seq_to_arr(seq):
    return np.array([input_symbols.get(x, 20) for x in seq])

def prepare_batch(seqs):
    seq_arr = [seq_to_arr(s) for s in seqs]
    seq_arr = pad_sequences(seq_arr, maxlen=2000, padding="post")
    seq_arr = to_categorical(seq_arr, num_classes=21)
    return seq_arr

def read_sequences(fn):
    if fn.endswith('.gz'):
        op = functools.partial(gzip.open, encoding='UTF-8')
    else:
        op = open

    with op(fn, 'rt') as fh:
        for r in Bio.SeqIO.parse(fh, "fasta"):
            # if importing from uniprot
            if r.id[2] == '|':
                _, rec_id, _ = r.id.split('|')
            else:
                rec_id = r.id
            seq = str(r.seq)
            yield rec_id, seq

def grouper(iterable, size=64):
    """Groups an iterable into size
    https://stackoverflow.com/a/8290514/2320823
    """
    sourceiter = iter(iterable)
    while True:
        batchiter = islice(sourceiter, size)
        try:
            yield chain([next(batchiter)], batchiter)
        except StopIteration:
            return

