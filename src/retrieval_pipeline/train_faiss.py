
import h5py
import numpy as np
from faiss_utils import FaissWrapper

# train core-100 matching stuffs
fb = h5py.File("/path/to/data/aser_embedding/sbert-all-MiniLM-L6-v2-core100-normed.hdf5", 'r') 

m_embeds = np.asarray(fb['embedding'])
m_words = fb['words']
print(m_embeds)

vector_dim = 384
n_centroids = 1024
code_size = 32
n_bits = 8
n_probe = 64
# devices=None means using all gpus, devices=0 means use gpu0
faiss_index = FaissWrapper(vector_dim, n_centroids, code_size, n_bits, n_probe, use_gpu=True, devices=None) 
faiss_index.train(m_embeds)
faiss_index.to_cpu()
faiss_index.save('/path/to/data/aser_retrieval_data/core_100_normed/faiss.index')