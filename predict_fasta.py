"""
Predicts labels from dspace given a fasta formatted file
"""

import json

import requests
import click

import numpy as  np
import pandas as pd

import sgidspace.inference
import sgidspace.load_outputs

from utils import read_sequences, grouper, prepare_batch

class InferenceEngineSlim(sgidspace.inference.InferenceEngine):
    def __init__(self):
        self.onehot_count = 5
        self.multihot_thresh = 0.1
        self.oprecision = 3
        self.eprecision = 6
        self.skip_tasks = ['embedding_autoencoder']
        self.vector_form = False

        self.outputs = sgidspace.load_outputs.load_outputs('outputs.txt')
        self.outputs = {o['name']: o for o in self.outputs}

    def _signif(self, x, n=3):
        x[np.isclose(x, 0)] = 0
        return np.around(x.astype(np.float64), n)

    def _filter_and_format_hot(self, output, probs, good_indices):
        """See InferenceEngine version for comments
        """
        good_labels = [output['class_labels'][i] for i in good_indices]
        good_probs = self._signif(probs[good_indices], self.oprecision)
        
        # ignore any zero labels
        good = pd.Series(good_probs, index=good_labels)
        good = good[~np.isclose(good, 0)]

        return good.to_dict()

    def generate(self, yhat):
        """
        Add labels and format predictions for a single batch of outputs
        """

        n_records = len(yhat[list(yhat.keys())[0]])
        records = [{}] * n_records

        for pred_name, preds in yhat.items():
            output_name = pred_name.split('/')[0]
            if output_name in self.skip_tasks:
                continue
            for rec_i, probs_i in enumerate(preds):
                probs_i = np.array(probs_i)

                output_meta = self.outputs[output_name]
                records[rec_i][output_name] = self._format_probs(output_meta, probs_i)
        return records
                
infer_sgi = InferenceEngineSlim()

def infer_batch(seqs, tfserver):
    """
    Returns 3d then 8d
    """
    seq_arr = prepare_batch(seqs)

    # reshape for request
    seq_arr = seq_arr.astype(int)
    seq_arr = seq_arr.tolist()
    seq_arr = [s for s in seq_arr]

    # send data to tf server
    payload = {
        "inputs": {
            "input_seq_batch": seq_arr
        }
    }
    r = requests.post(tfserver, json=payload)
    pred = json.loads(r.content.decode('utf-8'))
    pred = infer_sgi.generate(pred['outputs'])

    return pred

@click.command()
@click.argument('fasta_file')
@click.option('--tfhost', default='localhost:8501')
@click.option('--tfpath', default='/v1/models/dspace:predict')
def import_fasta(fasta_file, tfhost, tfpath):

    tfserver = 'http://{}{}'.format(tfhost, tfpath)
    seqiter = read_sequences(fasta_file)

    prefix = fasta_file.replace('.fasta.gz', '')

    for batch in grouper(seqiter):
        ids, seqs = zip(*batch)
        preds = infer_batch(seqs, tfserver)
        for i, s, p in zip(ids, seqs, preds):
            p['id'] = i
            p['seq'] = s
            print(p)
        

    return


if __name__ == '__main__':
    import_fasta()

