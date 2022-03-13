# Vision Transformer (ViT) usado para Calibração Automática de Parâmetros Intrínsecos de Câmeras

## Table of contents

- [Requirements](#requirements)
- [Dataset generation](#dataset-generation)
- [Training DeepCalib](#training-deepcalib)
- [Undistortion](#undistortion)
- [Citation](#citation)


## Requirements
- Python 3.7
- Keras 2.8
- TensorFlow 2.8
- OpenCV 4.5.5

## Dataset generation
There is a code for the whole data generation pipeline. First you have to download sun360 dataset using Google drive [link](https://drive.google.com/drive/folders/1ooaYwvNuFd-iEEcmOQHpLunJEmo7b4NM). Then, use the code provided to generate your continuous dataset. Please, do not forget to [cite](https://scholar.google.co.kr/scholar?hl=en&as_sdt=0%2C5&as_vis=1&q=recognizing+scene+viewpoint+using+panoramic+place+representation&btnG=#d=gs_cit&u=%2Fscholar%3Fq%3Dinfo%3ARJsOQOkTaMEJ%3Ascholar.google.com%2F%26output%3Dcite%26scirp%3D0%26hl%3Den) the paper describing sun360 dataset.

## Training DeepCalib
To train choose you network: SingleNet, ResNet-50 or Vision Transformer (ViT). All the training codes are available in this [folder](https://github.com/arianyfranca01/calibration_of_intrinsic_camera_parameters/tree/main/network_training).

#### Undistortion
There is a folder whit MATLAB code to undistort multiple images from .txt file. The format of the .txt file is the following: 1st column contains `path to the image`, 2nd column is `focal length`, 3rd column is `distortion parameter`. Each row corresponds to a single image. With a simple modification you can use it on a single image by giving direct path to it and predicted parameters. However, you need to change only `undist_from_txt.m` file, not the `undistSphIm.m`.
It is one way to qualitatively assess the accuracy of predicted parameters is to use those to undistort images that were used to predict the parameters. [Undistoriton](https://github.com/arianyfranca01/calibration_of_intrinsic_camera_parameters/tree/main/undistortion)

## Citation
```
@inproceedings{bogdan2018deepcalib,
  title={DeepCalib: a deep learning approach for automatic intrinsic calibration of wide field-of-view cameras},
  author={Bogdan, Oleksandr and Eckstein, Viktor and Rameau, Francois and Bazin, Jean-Charles},
  booktitle={Proceedings of the 15th ACM SIGGRAPH European Conference on Visual Media Production},
  year={2018}
}

@inproceedings{xiao2012recognizing,
  title={Recognizing scene viewpoint using panoramic place representation},
  author={Xiao, Jianxiong and Ehinger, Krista A and Oliva, Aude and Torralba, Antonio},
  booktitle={2012 IEEE Conference on Computer Vision and Pattern Recognition},
  year={2012},
}
```
