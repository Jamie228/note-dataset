# Undergraduate Major Project
This repository contains key scripts used to develop my Undergraduate Major Project, as well as the complete dataset used for object detection training.

## About This Project
The goal of this project is to produce an application capable of basic OMR (Optical Music Recognition). The application will be deployed on Android and should be able to detect, as a proof-of-concept, 8 different crotchet (quarter) notes denoting the notes C3 to C4 (in treble clef).

## Python Scripts
Below is a list of Python scripts (written in Google Colab) that have been used in the project.

- [create_pascal_voc.ipynb](create_pascal_voc.ipynb): A script written to produce Pascal-VOC style XML files for each image in the dataset from the CSV containing all bounding boxes for the dataset.

## The Dataset
The dataset was originally put together for object detection, but may also be used for image classification if that's the intention. A CSV file is provided that contains all 3020 bounding boxes for the dataset. The CSV file does not have a header row, but the columns are as follows:

``` class_name, xmin, ymin, xmax, ymax, filename, width, height ```

### Class Names
Dataset class names can be used to classify images as note names or crotchet notes in general. If classifying as crotchet notes, the only available class is ```note```, and there are 3020 bounding boxes in total for this class.

If classifying as note names, there are 8 classes in total: ```LC, D, E, F, G, A, B, HC```, where ```LC``` refers to the note C3 and ```HC``` refers to the note C4. Across the images, these classes are somewhat more irregular and a breakdown of bounding boxes can be found below:
- ```LC```: 123 bounding boxes
- ```D```: 170 bounding boxes
- ```E```: 224 bounding boxes
- ```F```: 265 bounding boxes
- ```G```: 789 bounding boxes
- ```A```: 636 bounding boxes
- ```B```: 509 bounding boxes
- ```HC```: 304 bounding boxes

### The Images
There are a total of 18 images used for the dataset, with data augmentation performed on each one to produce 31 unique images. The 18 images are named as follows:

```LC, D, E, F, G, A, B, HC, XA, XB, XC, XD, XE, XF, XG, XH, XI, XJ```

The original image will be named, for example ```XD.jpeg``` and augmentations produce files with a suffix. For example ```A_12.jpeg``` or ```XF_30.jpeg```.

Images ```LC``` through to ```HC``` are of one note corresponding to the image's name and have one bounding box associated with it. The images ```XA``` through to ```XJ``` are images containing multiple different notes, and therefore have multiple bounding boxes per image.
