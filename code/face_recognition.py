from facenet_pytorch import MTCNN, InceptionResnetV1
from concurrent.futures import ProcessPoolExecutor
from memory import Memory
import torch
from PIL import Image
from datetime import datetime


def handle_rotate(img: Image.Image):
    """
    Some image may include EXIF (Exchangeable Image File Format) metadata.
    One of the effect is the autorotation of the image, which cannot be automatically detect by the program.
    The models used in this program is sensitive to the rotation of the images.
    In that case, this function can read the EXIF and do the correct rotation.
    :param img: The image to be rotated.
    :return: The rotated image.
    """
    if not isinstance(img, Image.Image):
        raise TypeError(f'Image type {type(img)} not supported.')

    exif = img.getexif()

    if exif is not None:
        for tag, value in exif.items():
            if tag == 274:
                if value == 3:
                    img = img.rotate(180, expand=True)
                elif value == 6:
                    img = img.rotate(270, expand=True)
                elif value == 8:
                    img = img.rotate(90, expand=True)
    return img


def handle_rotation_multithreading(img_list: list[Image.Image]):
    with ProcessPoolExecutor() as executor:
        rotated_img_list = list(executor.map(handle_rotate, img_list))
    return rotated_img_list


def face_recognition(memory: Memory, device: torch.device, mtcnn: MTCNN, resnet: InceptionResnetV1, image: Image.Image,
                     multi_face: bool = False, save_detections: bool = False, save_detections_path: str = "../record/",
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
        faces = mtcnn(image, save_path=save_detections_path+datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    else:
        faces = mtcnn(image)

    if faces is None:
        return None

    faces = faces.to(device)

    if not multi_face:
        faces = torch.unsqueeze(faces, 0)

    embeddings = resnet(faces)

    distances = torch.cdist(embeddings, memory.get_embeddings(device), p=2)
    min_distances, min_indices = torch.min(distances, dim=1)
    min_indices[min_distances > threshold] = -1

    return min_indices


if __name__ == '__main__':
    img = Image.open('../test_pic/windy_on_train.jpg')
    img = handle_rotate(img)
    res = face_recognition(Memory.load_memory(), torch.device('cuda'), MTCNN(device=torch.device('cuda')),
                     InceptionResnetV1(pretrained='vggface2'), img,
                     multi_face=True)

    print(res)

