from concurrent.futures import ProcessPoolExecutor
from PIL import Image, ImageOps
import os


def handle_rotation(img: Image.Image):
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

    return ImageOps.exif_transpose(img)


def handle_rotation_multithreading(img_list: list[Image.Image]):
    with ProcessPoolExecutor() as executor:
        rotated_img_list = list(executor.map(handle_rotation, img_list))
    return rotated_img_list


def load_image_with_exif(img: Image.Image):
    return ImageOps.exif_transpose(img)


def load_image(file_path: str, rotation: bool = False):
    img = Image.open(file_path).convert('RGB')
    if rotation:
        img = handle_rotation(img)
    return img


def resize_images(input_folder, output_folder, size, rotation=False):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith(('jpg', 'jpeg', 'png')):
            img_path = os.path.join(input_folder, filename)
            img = load_image(img_path, rotation=rotation)

            img_resized = img.resize(size, Image.LANCZOS)

            output_path = os.path.join(output_folder, filename)
            img_resized.save(output_path)


def current_folder():
    return os.path.dirname(os.path.abspath(__file__))
