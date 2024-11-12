import os.path
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from PIL import Image
from datetime import datetime
from torch.utils.data import DataLoader

from memory import Memory
from images_dataset import ImageDataset
from process import current_folder


def face_recognition(memory: Memory, device: torch.device, mtcnn: MTCNN, resnet: InceptionResnetV1, image: Image.Image,
                     multi_face: bool = False, save_detections: bool = False,
                     save_detections_path: str = os.path.join(current_folder(), "../record/"),
                     threshold: float = 0.85):
    """
    Recognize the faces in an image.
    :param memory: The memory of the program.
    :param device: The device to use.
    :param mtcnn: The mtcnn model.
    :param resnet: The resnet model.
    :param image: The image to recognize. Only one image is permitted.
    :param multi_face: Whether to use multi-face recognition.
    :param save_detections: Whether to save the detected faces.
    :param save_detections_path: The path to save the detected faces.
    :param threshold: The threshold for face recognition. If the distance between the two faces is greater than the
                      threshold, the face is considered not possibly be this person. If cannot find any person for
                      the face, it is considered a stranger, or 'NOBODY'.
    :return: return a tensor of face recognition results. If n faces is detected, a tensor of face recognition results
             is a tensor whose length is n. The number in the tensor refers to the index of the person. For 'NOBODY',
             the number is -1.
    """
    if not memory.is_initialized():
        raise Exception('Memory is not initialized.')

    if not isinstance(image, Image.Image):
        raise Exception('Image is not a PIL Image.')

    if mtcnn.device != device:
        raise Exception("The device is different than the mtcnn device.")

    image = image.convert('RGB')

    mtcnn = mtcnn.eval()
    resnet = resnet.eval().to(device)

    mtcnn.keep_all = multi_face

    if save_detections:
        path = os.path.join(save_detections_path, datetime.now().strftime("%Y-%m-%d-%H-%M-%S")+'.jpg')
        faces = mtcnn(image, save_path=path)
    else:
        faces = mtcnn(image)

    if faces is None:
        return []

    faces = faces.to(device)

    if not multi_face:
        faces = torch.unsqueeze(faces, 0)

    embeddings = resnet(faces)

    distances = torch.cdist(embeddings, memory.get_embeddings(device), p=2)
    min_distances, min_indices = torch.min(distances, dim=1)
    min_indices[min_distances > threshold] = -1

    return memory.get_names(min_indices)


def multi_faces_recognition(memory: Memory, device: torch.device, mtcnn: MTCNN, resnet: InceptionResnetV1,
                            images_dataset: ImageDataset, save_detections: bool = False,
                            save_detections_path: str = os.path.join(current_folder(), "../record/"),
                            threshold: float = 0.85):
    if not memory.is_initialized():
        raise Exception('Memory is not initialized.')

    if mtcnn.device != device:
        raise Exception("The device is different than the mtcnn device.")
    dataloader = DataLoader(images_dataset, batch_size=16, collate_fn=collate_fn)

    mtcnn = mtcnn.eval()
    resnet = resnet.eval().to(device)

    mtcnn.keep_all = False

    all_names = []
    all_classes = []

    for images, names in dataloader:
        if save_detections:
            path = [os.path.join(save_detections_path, datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + f'{name}.jpg')
                    for name in names]
            faces = mtcnn(images, save_path=path)
        else:
            faces = mtcnn(images)

        faces = torch.stack(faces, dim=0).to(device)

        embeddings = resnet(faces)
        distances = torch.cdist(embeddings, memory.get_embeddings(device), p=2)
        min_distances, min_indices = torch.min(distances, dim=1)
        min_indices[min_distances > threshold] = -1
        all_names.extend(names)
        all_classes.extend(memory.get_names(min_indices))

    return all_names, all_classes


def collate_fn(batch):
    images = [item[0] for item in batch]
    filenames = [item[1] for item in batch]
    return images, filenames


if __name__ == '__main__':
    img = Image.open(os.path.join(current_folder(), '../test_pic/windy_on_train.jpg'))
    from process import handle_rotation
    img = handle_rotation(img)
    res = face_recognition(Memory.load_memory(), torch.device('cuda'), MTCNN(device=torch.device('cuda')),
                     InceptionResnetV1(pretrained='vggface2'), img,
                     multi_face=False, save_detections=False)

    print(res)

