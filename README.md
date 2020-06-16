# Learning Human-Object Interaction Detection using Interaction Points

Created by Tiancai Wang, Tong Yang, Martin Danelljan, Fahad Shahbaz Khan, Xiangyu Zhang, Jian Sun

Link for our paper: [arxiv](https://arxiv.org/abs/2003.14023) and [CVPR2020](http://openaccess.thecvf.com/content_CVPR_2020/html/Wang_Learning_Human-Object_Interaction_Detection_Using_Interaction_Points_CVPR_2020_paper.html)

### Introduction
Understanding interactions between humans and objects is one of the fundamental problems in visual classification and an essential step  towards detailed scene understand-ing. Human-object interaction(HOI) detection strives to localize both the human and an object as well as the identification of complex interactions between them. Most existing HOI detection approaches are instance-centric where interactions between all possible human-object pairs are predicted based on appearance features and coarse spatial information. We argue that appearance features aloneare insufficient to capture complex human-object interactions. In this paper, we therefore propose a novel fully-convolutional approach that directly detects the interactions between human-object pairs. Our network predicts interaction points, which directly localize and classify the  interaction. Paired with the densely predicted interaction vectors, the interactions are associated with human and object detections to obtain final predictions. To the  best  of  ourknowledge, we are the first to propose an approach whereHOI detection is posed as a keypoint detection and group-ing problem. Experiments are performed on two popularbenchmarks: V-COCO and HICO-DET. Our approach sets a new state-of-the-art on both datasets. 

## Installation
- Clone this repository. This repository is mainly based on [CenterNet](https://github.com/xingyizhou/CenterNet) and [iCAN](https://github.com/vt-vl-lab/iCAN).

```Shell
    IPNet_ROOT=/path/to/clone/IPNet
    git clone https://github.com/vaesl/IP-Net $IPNet_ROOT
```
- The code was tested on Ubuntu 18.04, with [Anaconda](https://www.anaconda.com/download) Python 3.6 and [PyTorch]((http://pytorch.org/)) v1.0.1. 
NVIDIA GPUs are needed for testing. After install Anaconda, create a new conda environment, activate the environment and install pytorch1.0.1.

```Shell
    conda create -n IPNet python=3.6
    source activate IPNet
    conda install pytorch=1.0.1 torchvision -c pytorch
```

- Install the requirements. 
```Shell
    pip3 install -r requirements.txt
```
- Compiling Center Pooling Layers.
```Shell
    cd IPNet_ROOT/src/lib/models/networks/py_utils/_cpools/
    python setup.py install --user
```

- Install [COCOAPI](https://github.com/cocodataset/cocoapi):

    ~~~
    # COCOAPI=/path/to/clone/cocoapi
    git clone https://github.com/cocodataset/cocoapi.git $COCOAPI
    cd $COCOAPI/PythonAPI
    make
    python setup.py install --user
    ~~~

## Download
To evaluate the performance reported in the paper, V-COCO and HICO-DET dataset as well as our trained models need to be downloaded.

### V-COCO and HICO-DET Datasets
Download datasets and setup evaluation and API, please follow [iCAN](https://github.com/vt-vl-lab/iCAN).

### Trained Models
Please access [Google Driver](https://drive.google.com/file/d/1stBqpTncUFfl-naKn4NONRmC-89jtdyh/view?usp=sharing) 
to obtain our trained models for V-COCO and put the models into corresponding directory(e.g. '~/weights/V-COCO/'). 
Note that we only release models of V-COCO  for the time being. 

## Evaluation
To check the performance reported in the paper, just simply run:

```Shell
python3 test_HOI.py ctdet --exp_id coco_hg --fix_res --arch hourglass --flip_test  --load_model /path/to/model/weights
```

## Citation
Please cite our paper in your publications if it helps your research:

    @article{Wang2020IPNet,
        title = {Learning Human-Object Interaction Detection using Interaction Points},
        author = {Tiancai Wang, Tong Yang, Martin Danelljan, Fahad Shahbaz Khan, Xiangyu Zhang, Jian Sun},
        booktitle = {CVPR},
        year = {2020}
    }
