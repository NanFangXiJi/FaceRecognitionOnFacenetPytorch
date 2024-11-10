import os
import torch


class Memory:
    MEMORY_PATH = '../data/faces_memory.mpt'

    def __init__(self, save_detected=False):
        self.initialized = False
        self.idx_to_class = None
        self.class_to_idx = None
        self.embeddings = None

        self.save_detected = save_detected

    def initialize(self, class_to_idx, embeddings):
        self.class_to_idx = class_to_idx.copy()
        self.idx_to_class = {i: c for c, i in class_to_idx.items()}
        self.embeddings = embeddings.to(torch.device('cpu'))
        self.save()

    def save(self):
        torch.save(self, self.MEMORY_PATH)

    @staticmethod
    def load_memory():
        if os.path.exists(Memory.MEMORY_PATH):
            return torch.load(Memory.MEMORY_PATH)
        else:
            return Memory()
