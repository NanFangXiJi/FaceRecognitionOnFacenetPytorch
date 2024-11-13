# 基于FaceNet-Pytorch的人脸识别

To read English version readme, click [readme](readme.md).

---

## 支持的图片格式
支持在图像中进行人脸检测。  
支持的格式包括JPG、PNG和JPEG。  
不保证支持其他格式。

---

## 使用指南
运行 `python code/main.py [命令]` 启动程序。

### **初始化数据集**  
该程序预处理已知人脸的数据集。每个人脸的嵌入信息和其他状态将被保存在 `data/faces_memory.mpt` 文件中。初始化命令如下：

```
init [-f | --filepath filepath] [-r | --rotation] [-sg | --single] [-c | --cpu]
```

- `filepath` 是您自己的图像数据集的目录路径。
- 如果没有提供 `filepath`，则默认使用目录 `data/faces_memory`。

***注意！*** 数据集目录的结构应如下所示：

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

***注意！*** 不应使用 `'Nobody'` 作为类名，因为它用于标记陌生人。每个人的图像应放置在自己的目录中。

- `-r | --rotation` 处理EXIF元数据中的旋转。有些图像，特别是由相机或智能手机拍摄的图像，可能会有EXIF元数据，该数据保存了正确的角度，但需要根据该角度旋转矩阵。如果没有旋转，可能会导致程序无法正确检测人脸。除非您确定图像中没有EXIF旋转数据，否则应使用此参数。
- `-sg | --single` 只选择每个人的一张图像，而不是计算所有图像的特征并保存它们的平均值。如果某些图像质量较差，可能会显著影响平均特征，建议使用此选项。
- `-c | --cpu` 使用CPU进行处理。如果没有使用此参数，程序将在没有CUDA的情况下默认使用CPU。

您的数据集应该尽可能确保每张图片只有一个人脸，或者至少每张图像中最大的人脸应属于正确的类。

只要运行了 `init`，后续特征值将自动加载以供预测使用。

数据集初始化完成后，可以进行人脸识别。

### **人脸识别**

要识别单张图像，请使用以下命令：

```
rec filepath [-m | --multi-faces] [-r | --rotation] [-sf | --save-faces [filepath]] [-c | --cpu] [-th | --threshold threshold]
```

该命令用于检测单张图像。`filepath` 参数是必需的。

- `-m | --multi-faces` 允许在一张图像中检测多个人脸。
- `-sf | --save-faces [filepath]` 保存所有检测到的人脸。检测到的人脸将保存在 `record/` 目录或您指定的目录中。
- `-th | --threshold threshold` 设置特征距离的阈值。当两个特征的距离小于此阈值时，它们将被认为是同一个人。否则，它们将被标记为 'NOBODY'。建议使用默认值，合理的阈值通常在0.6到0.9之间。

### **识别目录中的所有图像**

***注意！*** 此模式不支持在一张图像中检测多个人脸。

```
rec_all filepath [-ss | --same-size] [-r | --rotation] [-sf | --save-faces [filepath]] [-c | --cpu] [-th | --threshold threshold]
```

- `-ss | --same-size` 表示所有图像的大小相同，并且每张图像必须至少包含一张人脸。如果您确定所有图像的大小相同，强烈建议使用此选项，这将显著加快人脸识别速度。

### **调整目录中所有图像的大小**

```
resize height width input_dir output_dir [-r | --rotation]
```

如果您的图像有相似的长宽比，您可能希望将它们调整为统一的大小，以便在 `rec_all` 中使用 `-ss`。此命令将把 `input_dir` 中的所有图像调整为指定的尺寸（`height` 和 `width`），并将它们保存在 `output_dir` 中。

如果您使用 `-r | --rotation` 参数处理图像，则运行 `rec` 或 `rec_all` 时无需再次旋转图像。

请小心，调整大小可能导致无法成功检测到人脸。
