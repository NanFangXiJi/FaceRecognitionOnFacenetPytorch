# Face Recognition On FaceNet-Pytorch

---

## Supported image format
Support to detect faces in images. 
Supported format includes `jpg`, `png`, `jpeg`, `bmp`. 
Support on other format is not guaranteed.

---

## Guidance
Run the `main.py` to start the program.

Overall structure is
```
python code/main.py [command]
```

For the `[command]`, it is explained below.

### **Initialize the dataset.** 
This program needs to preprocess the dataset of the known faces.
The embedding of each face, as well as other status will
be saved in the file `data/faces_memory.mpt`. The command
to initialize is
```
init [-f | --filepath filepath] [-r | --rotation] [-sg | --single] [-c | --cpu]
```
`filepath` is the directory path of your own images dataset.

If no `filepath` is given, the default directory `data/faces_memory`
is to be read.

***Attention!*** The structure of the dataset directory should be like
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
***Attention!*** No classname should be 'Nobody', 
because this is the name we label strangers.

That means, images of one person should have their own directory.

`-r | --rotation` is process the rotation in *EXIF* metadata. 

Sometimes there are EXIF metadata in your images, 
particularly for those pictures taken by your camera or smartphone. They will show the correct angle in most software, 
but it is saved as a spun matrix, which cannot be detected by our program successfully.

Use this parameter unless you are convinced that there are no EXIF rotation in your images.

`-sg | --single` is the parameters that set the features to select only one image for each person,
rather than calculate all embeddings of each image and save their average.

It is suggested only when you believe that some of the images are low quality images, 
which can cause severe damage to the average embedding.

`-c | --cpu` means using cpu to run. Without this parameter, the program uses cpu only when cuda is not accessible.

Your dataset is expected to have only one face in each image, or at least the face for the class is the biggest face in the picture.

Once you run the `init`, the embeddings value will be automatically loaded for predict.

### **Recognize the face**
Once it is initialized, recognition is ready.

For one image in a time, use this command.
```
rec filepath [-m | --multi-faces] [-r | --rotation] [-sf | --save-faces [filepath]] [-c | --cpu] [-th | --threshold threshold]
```
This command is designed for detect one image at a time.
The `filepath` is necessary.

`-m | --multiface` means allow to detect more than one face at a time.

`-sf | --saveface [filepath]` means save all detected faces. Detected faces are saved in `record/` or the directory you give.

`-th | --threshold threshold` gives the threshold of the embeddings' distances. 
When two embeddings have lower distance than threshold, they are possibly considered to be the same person.
Otherwise, they are considered 'NOBODY'.
