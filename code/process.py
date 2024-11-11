from concurrent.futures import ProcessPoolExecutor
from PIL import Image, ImageOps


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
