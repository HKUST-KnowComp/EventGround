# coding: utf-8

import faiss

class FaissWrapper:
    def __init__(self, vector_dim=384, n_centroids=1024, code_size=32, n_bits=8, n_probe=64, use_gpu=False, devices=None):
        '''
            n_centroids <int>:  The feature space is partitioned into n_centroids cells.
            code_size <int>:    code_size (m) is typically a power of 2 between 4 and 64, and vector_dim (d) should be a    multiple of m
            n_bits <int>:       The number of bits n_bits must be equal to 8, 12 or 16.
            n_probe <int>:      The nprobe is specified at query time (useful for measuring trade-offs between speed and accuracy). Setting nprobe = nlist gives the same result as the brute-force search (but slower).

            use_gpu <bool>:     Whether to train the faiss index on gpu or not (default to False).
        '''
        self.use_gpu = use_gpu
        self.devices = devices
        self.n_probe = n_probe

        quantizer = faiss.IndexFlatL2(vector_dim)
        index_cpu = faiss.IndexIVFPQ(quantizer, vector_dim, n_centroids, code_size, n_bits)
        self.index = index_cpu

        if self.use_gpu:
            self.to_gpu(self.devices)

    def train(self, matrix):
        ''' Train the faiss index on a given matrix of shape [n, vector_dim].
        '''
        assert not self.index.is_trained
        self.index.train(matrix)
        self.index.add(matrix)

    def search(self, query, top_k=10):
        return self.index.search(query, top_k)

    def set_nprobe(self, n_probe):
        assert isinstance(n_probe, int) and n_probe > 0
        self.index.nprobe = n_probe
        self.n_probe = n_probe

    def to_cpu(self):
        ''' Copy the gpu faiss model to cpu, and change to use the cpu model.
        '''
        assert self.use_gpu is True
        self.index = faiss.index_gpu_to_cpu(self.index)


    def to_gpu(self, devices=None):
        ''' Copy the cpu faiss model to gpu, and change to use the gpu model. By default use all gpus.
        '''
        assert self.use_gpu is True

        if devices is None:
            self.index = faiss.index_cpu_to_all_gpus(self.index)
        else:
            if not isinstance(devices, int):
                devices = 0
            self.index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), devices, self.index)

    def save(self, filename='faiss.index'):
        try:
            faiss.write_index(self.index, filename)
        except Exception as e:
            print(e)
            print('Note: this is probably because the current device is not cpu, try self.to_cpu() first.')

    def load(self, filename='faiss.index'):
        self.index = faiss.read_index(filename)