from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision import datasets
from collections import defaultdict
import torch
import os

from memory import Memory


def read_dataset(memory: Memory, device: torch.device, mtcnn: MTCNN, resnet: InceptionResnetV1, dataset_path: str = '../data/faces_memory',
                 only_one_picture: bool = False):
    """
    Read the dataset of known faces, generate their embeddings and save them in memory.
    :param memory: The memory of the program.
    :param device: The device to use.
    :param mtcnn: The mtcnn model.
    :param resnet: The resnet model.
    :param dataset_path: The path to the dataset.
    :param only_one_picture: Whether to only read one picture for generate embeddings.
    """

    if mtcnn.device != device:
        raise Exception("The device is different than the mtcnn device.")

    mtcnn = mtcnn.eval()
    resnet = resnet.eval().to(device)
    mtcnn.keep_all = False

    try:
        os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
        dataset = datasets.ImageFolder(dataset_path)
    except FileNotFoundError as e:
        raise Exception(f'Read {dataset_path} failed.')

    if len(dataset) == 0:
        raise Exception(f'No image found in {dataset_path}.')

    dataset.idx_to_class = {i: c for c, i in dataset.class_to_idx.items()}

    aligned = []
    names = []

    # possible improvement: if the images are guaranteed to be the same size, batch process is possible.
    for x, y in dataset:
        if only_one_picture and y in names:
            continue

        faces = mtcnn(x)
        if faces is not None:
            aligned.append(faces)
            names.append(dataset.idx_to_class[y])
        else:
            """
            picture y has no face detected.
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
    read_dataset(Memory(), torch.device('cuda'), MTCNN(device=torch.device('cuda')), InceptionResnetV1(pretrained='vggface2'))
