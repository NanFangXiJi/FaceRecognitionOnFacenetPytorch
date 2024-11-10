from facenet_pytorch import MTCNN, InceptionResnetV1
from memory import Memory
import torch
from PIL import Image


def face_recognition(memory: Memory, device: torch.device, mtcnn: MTCNN, resnet: InceptionResnetV1, image: Image.Image,
                     multi_face: bool = False, save_detections: bool = False):
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

    faces = mtcnn(image)

    if faces is None:
        return None

    # TODO
