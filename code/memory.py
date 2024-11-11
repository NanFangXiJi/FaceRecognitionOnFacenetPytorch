import os
import torch


class Memory:
    MEMORY_PATH = '../data/faces_memory.mpt'
    NOBODY = 'Nobody'

    def __init__(self):
        self.initialized = False
        self.idx_to_class = None
        self.class_to_idx = None
        self.embeddings = None

    def initialize(self, class_to_idx, embeddings, device):
        self.class_to_idx = class_to_idx.copy()
        self.class_to_idx[self.NOBODY] = -1
        self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}

        self.embeddings = embeddings.to(device)
        self.initialized = True
        self.save()

    def save(self):
        os.makedirs(os.path.dirname(self.MEMORY_PATH), exist_ok=True)
        torch.save(self, self.MEMORY_PATH)

    def person_num(self):
        return len(self.class_to_idx) - 1

    def is_initialized(self):
        return self.initialized

    def save_detected(self, if_save_detected):
        self.save_detected = if_save_detected

    def get_embeddings(self, device=torch.device('cpu')):
        return self.embeddings.to(device)

    def get_names(self, indices: torch.Tensor):
        names = []
        for elem in indices:
            names.append(self.idx_to_class[elem.item()])
        return names

    @staticmethod
    def load_memory():
        if os.path.exists(Memory.MEMORY_PATH):
            return torch.load(Memory.MEMORY_PATH)
        else:
            return Memory()
