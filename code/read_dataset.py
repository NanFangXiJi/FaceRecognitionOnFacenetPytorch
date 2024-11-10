from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision import datasets
from collections import defaultdict
import torch

from memory import Memory


def read_dataset(memory: Memory, device: torch.device, dataset_path: str = '../data/faces_memory'):
    """
    Read the dataset of known faces, generate their embeddings and save them in memory.
    :param memory: The memory of the program.
    :param device: The device to use.
    :param dataset_path: The path to the dataset.
    """
    mtcnn = MTCNN(device=device).eval()
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    dataset = datasets.ImageFolder(dataset_path)
    dataset.idx_to_class = {i: c for c, i in dataset.class_to_idx.items()}

    aligned = []
    names = []

    # possible improvement: if the images are guaranteed to be the same size, batch process is possible.
    for x, y in dataset:
        x_aligned = mtcnn(x)
        if x_aligned is not None:
            aligned.append(x_aligned)
            names.append(dataset.idx_to_class[y])
        else:
            """
            dataset.idx_to_class[y] has no faces detected.
            """

    aligned = torch.stack(aligned).to(device)
    embeddings = resnet(aligned).detach().cpu()

    class_to_idx = dict()
    class_to_features = defaultdict(list)
    for i, class_name in enumerate(names):
        class_to_features[class_name].append(embeddings[i])
        class_to_idx[class_name] = i

    class_features = []
    for class_name, features in class_to_features.items():
        class_tensor = torch.stack(features)
        class_mean_feature = class_tensor.mean(dim=0)
        class_features.append(class_mean_feature)

    embeddings = torch.stack(class_features)

    memory.initialize(class_to_idx, embeddings)


if __name__ == '__main__':
    read_dataset(Memory(), torch.device('cuda'))
