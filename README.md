# Vision Transformer (ViT) usado para Calibração Automática de Parâmetros Intrínsecos de Câmeras

## Table of contents

- [Requirements](#requirements)
- [Dataset generation](#dataset-generation)
- [Training DeepCalib](#training-deepcalib)
- [Camera calibraition](#camera-calibration)
- [Notes](#notes)
  - [Different architectures](#different-architectures)
  - [Weights](#weights)
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
To train choose you network: SingleNet, ResNet-50 or Vision Transformer (ViT). All the training codes are available in this [folder](https://github.com/alexvbogdan/DeepCalib/tree/master/network_training).

## Camera Calibration
To infer distortion parameter and focal length of a given camera we take a short video, extract the frames and run the prediction on all of them. After that, we take the mean or the median of predicted values and use that as a final result. However, in a slight modification you can use them for a single image prediction as well. Below you can see some of the results of image rectification using parameters obtained from single image calibration. ![Results](https://github.com/alexvbogdan/DeepCalib/blob/master/Results.png)
In [prediction folder](https://github.com/alexvbogdan/DeepCalib/tree/master/prediction) we have the codes for all the networks except for `SeqNet` regression because the weights for this architecture are currently unavailable. We uploaded a simple python script for frame extraction from video sequence.

## Notes

#### Different architectures
For detailed information refer to the `Section 4.2` of [our paper](https://drive.google.com/file/d/1pZgR3wNS6Mvb87W0ixOHmEVV6tcI8d50/view). In short, `SingleNet` (a) is the best network for predicting focal length and distortion parameter in terms of accuracy. In addition, since it is a single network contrary to `DualNet` (b) and `Seqnet` (c), it is computationally cheaper to use the former. ![DeepCalib architectures](https://github.com/alexvbogdan/DeepCalib/blob/master/DeepCalib_architectures.png)

#### Weights
The weights for our networks can be found [here](https://drive.google.com/file/d/1TYZn-f2z7O0hp_IZnNfZ06ExgU9ii70T/view). We recommend to use `SingleNet` since we experimentally confirmed it outperforms the other ones. The regression weights for `SeqNet` are currently unavailable, although you can train your own.

#### Undistortion
One way to qualitatively assess the accuracy of predicted parameters is to use those to undistort images that were used to predict the parameters. [Undistoriton](https://github.com/alexvbogdan/DeepCalib/tree/master/undistortion) folder contains MATLAB code to undistort multiple images from .txt file. The format of the .txt file is the following: 1st column contains `path to the image`, 2nd column is `focal length`, 3rd column is `distortion parameter`. Each row corresponds to a single image. With a simple modification you can use it on a single image by giving direct path to it and predicted parameters. However, you need to change only `undist_from_txt.m` file, not the `undistSphIm.m`.

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
