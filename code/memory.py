import os
import torch


class Memory:
    MEMORY_PATH = '../data/faces_memory.mpt'

    def __init__(self):
        self.initialized = False
        self.idx_to_class = None
        self.class_to_idx = None
        self.embeddings = None

    def initialize(self, class_to_idx, embeddings):
        self.class_to_idx = class_to_idx.copy()
        self.idx_to_class = {i: c for c, i in class_to_idx.items()}
        self.embeddings = embeddings.to(torch.device('cpu'))
        self.save()
        self.initialized = True

    def save(self):
        os.makedirs(os.path.dirname(self.MEMORY_PATH), exist_ok=True)
        torch.save(self, self.MEMORY_PATH)

    def person_num(self):
        return len(self.class_to_idx)

    def is_initialized(self):
        return self.initialized

    def save_detected(self, if_save_detected):
        self.save_detected = if_save_detected

    @staticmethod
    def load_memory():
        if os.path.exists(Memory.MEMORY_PATH):
            return torch.load(Memory.MEMORY_PATH)
        else:
            return Memory()
