# coding: utf-8
# Embed ASER with sentence transformers

import argparse
import h5py
import os
import numpy as np
import torch
import networkx as nx
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sbert_model", type=str, default='all-MiniLM-L6-v2', help="Sentence BERT model name")
    parser.add_argument("--kg_dir", type=str, default="/path/to/data/core_100_normed_poss.pickle", help="Path to ASER KG")
    parser.add_argument("--emb_dir", type=str, default="/path/to/data/aser_embedding", help="Path to ASER KG")
    parser.add_argument("--name", type=str, default="core100-normed", help="Distinct label for this embedding")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('Loading sentence transformer...')
    model = SentenceTransformer(args.sbert_model)
    model = model.to(device)

    print('Loading ASER...')
    aser = nx.read_gpickle(args.kg_dir)
    eventualities = list(aser.nodes)
    
    print('Loaded. # events:', len(eventualities))

    embeddings = []

    length = len(eventualities)
    print(eventualities[:10])

    # [P0] -> person, [P1] -> person
    events_for_encoding = eventualities
    print(events_for_encoding[:10])

    # length = 100
    print('Embedding...')
    for i in tqdm(range(0, length, args.batch_size)):
        batch = events_for_encoding[i: i+args.batch_size]
        embs = model.encode([x for x in batch], batch_size=len(batch))
        embeddings.append(embs)
    embeddings = np.concatenate(embeddings, axis=0)

    print('Finished embedding aser. Start saving to file...')

    os.makedirs(args.emb_dir, exist_ok=True)
    filename = os.path.join(args.emb_dir, f"sbert-{args.sbert_model}-{args.name}.hdf5")
    print(f'Saving to {filename}')
    with h5py.File(filename, "w") as f:
        f.create_dataset('words', data=eventualities)
        f.create_dataset('embedding', data=embeddings)
    print('Done.')

if __name__ == '__main__':
    main()