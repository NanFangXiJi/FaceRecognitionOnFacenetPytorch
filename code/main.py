import os
import argparse
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch

from memory import Memory
from read_dataset import read_dataset
from face_recognition import face_recognition, multi_faces_recognition
from process import load_image, resize_images, current_folder
from images_dataset import ImageDataset


def argparse_process():
    parser = argparse.ArgumentParser(description='Face Recognition')
    subparsers = parser.add_subparsers(dest="command")

    init_parser = subparsers.add_parser("init", help="Initialize the database")
    init_parser.add_argument("-f", "--filepath", type=str,
                             default=os.path.join(current_folder(), '../data/faces_memory'),
                             help="Path to the database")
    init_parser.add_argument("-r", "--rotation", action='store_true',
                             help="Handle the EXIF rotation")
    init_parser.add_argument("-sg", "--single", action='store_true',
                             help="Read only one picture for each class")
    init_parser.add_argument("-c", "--cpu", action='store_true', help="Use CPU")

    rec_parser = subparsers.add_parser("rec", help="Recognize one picture")
    rec_parser.add_argument("filepath", type=str, help="Path to the picture")
    rec_parser.add_argument("-m", "--multi-faces", action='store_true',
                            help="Read multiple faces in one picture")
    rec_parser.add_argument("-r", "--rotation", action='store_true',
                             help="Handle the EXIF rotation")
    rec_parser.add_argument("-sf", "--save-faces", type=str, nargs="?",
                            const=os.path.join(current_folder(), "../record"),
                            help="save detected faces")
    rec_parser.add_argument("-c", "--cpu", action='store_true', help="Use CPU")
    rec_parser.add_argument("-th", "--threshold", type=float, default=0.85,
                            help="Threshold for detecting faces")

    rec_all_parser = subparsers.add_parser("rec_all", help="Recognize all pictures in one directory")
    rec_all_parser.add_argument("filepath", type=str, help="Path to the directory")
    rec_all_parser.add_argument("-ss", "--same-size", action='store_true',
                                help="Use quicker recognition mode if all your pictures are same size")
    rec_all_parser.add_argument("-r", "--rotation", action='store_true',
                                help="Handle the EXIF rotation")
    rec_all_parser.add_argument("-sf", "--save-faces", type=str, nargs="?",
                                const=os.path.join(current_folder(), "../record"),
                                help="save detected faces")
    rec_all_parser.add_argument("-c", "--cpu", action='store_true', help="Use CPU")
    rec_all_parser.add_argument("-th", "--threshold", type=float, default=0.85,
                                help="Threshold for detecting faces")

    resize_parser = subparsers.add_parser("resize", help="Resize pictures in a directory")
    resize_parser.add_argument("width", type=int, help="Width of output picture")
    resize_parser.add_argument("height", type=int, help="Height of output picture")
    resize_parser.add_argument("input_directory", type=str, help="Path to the input directory")
    resize_parser.add_argument("output_directory", type=str, help="Path to the output directory")
    resize_parser.add_argument("-r", "--rotation", action='store_true',
                               help="Handle the EXIF rotation")

    return parser.parse_args()


def get_device(cpu=False):
    if cpu:
        return torch.device('cpu')
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def get_mtcnn(device):
    return MTCNN(device=device)


def get_resnet(device):
    return InceptionResnetV1(device=device, pretrained='vggface2')


if __name__ == '__main__':
    args = argparse_process()

    if args.command == "resize":
        try:
            resize_images(args.input_directory, args.output_directory, (args.width, args.height), args.rotation)
        except Exception as e:
            print("resize failed")
            print(e)
            exit(1)
        exit(0)

    try:
        memory = Memory.load_memory()
    except Exception as e:
        print("Memory load failed:")
        print(e)
        exit(2)

    if args.command == "init":
        device = get_device(args.cpu)
        mtcnn = get_mtcnn(device)
        resnet = get_resnet(device)
        dataset_path = args.filepath
        exif_rotation = args.rotation
        single_picture = args.single

        try:
            read_dataset(memory, device, mtcnn, resnet, dataset_path, single_picture, exif_rotation)
        except Exception as e:
            print("read dataset failed:")
            print(e)
            exit(3)

    if args.command == "rec":
        if not memory.is_initialized():
            print("Not initialized")
            exit(4)
        device = get_device(args.cpu)
        mtcnn = get_mtcnn(device)
        resnet = get_resnet(device)
        filepath = args.filepath
        rotation = args.rotation
        multi_faces = args.multi_faces
        save_faces = args.save_faces is not None
        save_faces_path = args.save_faces
        threshold = args.threshold

        if save_faces and not os.path.isdir(save_faces_path):
            print(f"Save faces path {save_faces_path} not exist or is not a directory.")
            exit(5)

        try:
            img = load_image(filepath, rotation)
        except Exception as e:
            print(f"Load image {filepath} failed:")
            print(e)
            exit(6)

        try:
            names = face_recognition(memory, device, mtcnn, resnet, img, multi_faces,
                                     save_faces, save_faces_path, threshold)
        except Exception as e:
            print("face recognition failed:")
            print(e)
            exit(7)

        print("detected face(s):")
        for i, name in enumerate(names):
            print(f"{i}: {name}")

    if args.command == "rec_all":
        if not memory.is_initialized():
            print("Not initialized")
            exit(8)
        device = get_device(args.cpu)
        mtcnn = get_mtcnn(device)
        resnet = get_resnet(device)
        filepath = args.filepath
        same_size = args.same_size
        rotation = args.rotation
        save_faces = args.save_faces is not None
        save_faces_path = args.save_faces
        threshold = args.threshold

        if not os.path.isdir(filepath):
            print(f"filepath {filepath} not exist or is not a directory")
            exit(9)

        if save_faces and not os.path.isdir(save_faces_path):
            print(f"Save faces path {save_faces_path} not exist or is not a directory")
            exit(10)

        try:
            images = ImageDataset(filepath, device, rotation)
        except Exception as e:
            print(f"load images failed:")
            print(e)
            exit(11)

        if len(images) == 0:
            print("No images")
            exit(12)

        if same_size:
            try:
                names, classes = multi_faces_recognition(memory, device, mtcnn, resnet, images,
                                                         save_faces, save_faces_path, threshold)
            except Exception as e:
                print("faces recognition failed:")
                print(e)
                exit(13)
        else:
            names = []
            classes = []

            for image, name in images:
                try:
                    cls = face_recognition(memory, device, mtcnn, resnet, image, False,
                                           save_faces, save_faces_path, threshold)
                except Exception as e:
                    print("faces recognition failed:")
                    print(e)
                    exit(14)
                if cls is not None and len(cls) > 0:
                    names.append(name)
                    classes.append(cls[0])

        for name, cls in zip(names, classes):
            print(f"{name}: {cls}")
