# Face Recognition On FaceNet-Pytorch

简体中文(Simplified Chinese)文档已提供，请点击[中文readme](readme_zh.md)。

---
## Package Installment
### Pillow
```
pip install Pillow
```
### Torch and Torchvision
You can find the installation instructions on their [official website](https://pytorch.org/get-started/locally/).

Select the correct operating system, Python version, and CUDA version (or choose the CPU version if you do not have CUDA or a GPU).
Then, use the command provided on the website to install the packages.

### FaceNet-Pytorch
```
pip install facenet_pytorch
# Pay attention, sometimes it might replace your Torch and Torchvision with their version altomatically.
```
Alternatively, you can download it from [their repository](https://github.com/timesler/facenet-pytorch/releases).

---

## Supported Image Formats
Supports face detection in images.  
Supported formats include JPG, PNG, and JPEG.  
Support for other formats is not guaranteed.

---

## Guidance
Run `python code/main.py [command]` to start the program.

For the `[command]`, it is explained below.
Details can be found by `-h | --help`.

### **Initialize the Dataset**  
This program preprocesses the dataset of known faces. 
The embedding of each face, along with other status information, 
will be saved in the file `data/faces_memory.mpt`. 
The command to initialize is:
```
init [-f | --filepath filepath] [-r | --rotation] [-sg | --single] [-c | --cpu]
```
- `filepath` is the directory path of your own image dataset.
- If no filepath is provided, the default directory `data/faces_memory` will be used.


***Attention!*** The structure of the dataset directory should be as follows:
```
filepath/
    class_1/
        img1.jpg
        img2.jpg
        ...
    class_2/
        img1.jpg
        img2.jpg
        ...
    ...
```
***Attention!*** The classname 'Nobody' should not be used, 
as it is reserved for labeling strangers. 
This means that images of each person should be placed in their own directory.

- `-r | --rotation` processes the rotation in EXIF metadata. 
Some images, particularly those taken by cameras or smartphones, 
may have EXIF metadata that saves the correct angle but as a rotated matrix. 
This may prevent the program from detecting faces correctly. 
Use this parameter unless you are certain there is no EXIF rotation in your images.

- `-sg | --single` selects only one image per person, 
rather than calculating embeddings for all images and saving their average. 
This option is suggested if some images are of low quality, 
as they may cause significant damage to the average embedding.

- `-c | --cpu` uses the CPU for processing. If this parameter is not used, 
the program will default to using the CPU when CUDA is not available.

Your dataset should contain only one face per image, 
or at least the largest face in each image should belong to the correct class.

Once you run the `init`, the embedding values will be automatically loaded for predict.

Once the dataset is initialized, face recognition is ready.

### **Face Recognition**  
For recognizing a single image, use the following command:
```
rec filepath [-m | --multi-faces] [-r | --rotation] [-sf | --save-faces [filepath]] [-c | --cpu] [-th | --threshold threshold]
```
This command is designed for detect one image at a time.
The `filepath` is required.

- `-m | --multi-faces` allows for detecting multiple faces in one image.

- `-sf | --save-faces [filepath]` saves all detected faces. 
Detected faces are saved in the `record/` directory or the directory you specify.

- `-th | --threshold threshold` sets the threshold for comparing embeddings' distances. 
When the distance between two embeddings is lower than the threshold, 
they are considered to be the same person. 
Otherwise, they are labeled as 'NOBODY'. 
The default value is recommended, with reasonable values typically ranging from 0.6 to 0.9.

### **Recognize All Images in a Directory**
***Attention!*** This mode does not support detecting multiple faces in one image.

```
rec_all filepath [-ss | --same-size] [-r | --rotation] [-sf | --save-faces [filepath]] [-c | --cpu] [-th | --threshold threshold]
```

- `-ss | --same-size` means all your images are the same size and must contain at least one human face. 
This is recommended if you are sure that all images meet these requirements, as it will speed up face recognition.

### **Resize All Images in a Directory**

```
resize height width input_dir output_dir [-r | --rotation]
```

If your images have a similar aspect ratio, you may want to resize them to a consistent size, so you can use `-ss` in `rec_all`. 
This command resizes all images in `input_dir` to the specified dimensions (`height` and `width`) and saves them in `output_dir`.

If you process the images with `-r | --rotation`, 
there will be no need to rotate them again when you run `rec` or `rec_all`.

Be careful, resizing might result in failure to detect faces successfully.
