Copyright (C) 2020 ETH Zurich, Switzerland. SPDX-License-Identifier: Apache-2.0. See LICENSE file for details.

Authors: Xiaying Wang, Michael Hersche

# An Accurate EEGNet-based Motor-Imagery Brain--Computer Interface for Low-Power Edge Computing

In this repository, we share the code for classifying MI data of the Physionet EEG Motor Movement/Imagery Dataset using EEGNet. 
For details, please refer to the papers below. 

If this code proves useful for your research, please cite
> Xiaying Wang, Michael Hersche, Batuhan Tömekce, Burak Kaya, Michele Magno, Luca Benini, "An Accurate EEGNet-based Motor-Imagery Brain--Computer Interface for Low-Power Edge Computing", in IEEE International Symposium on Medical Measurements and Applications (MEMEA), 2020.  
<!--DOI (preprint): [10.3929/ethz-b-000282732](https://www.research-collection.ethz.ch/handle/20.500.11850/282732). Available on [arXiv](https://arxiv.org/pdf/1808.05488). -->



#### Installing Dependencies
You will need a machine with a CUDA-enabled GPU and the Nvidia SDK installed to compile the CUDA kernels.
Further, we have used conda as a python package manager and exported the environment specifications to `dependency.yml`. 
You can recreate our environment by running 

```
conda env create -f dependency.yml -n mybciEnv 
```
Make sure to activate the environment before running any code. 

If get a batchnormalization error when running the code, follow the instructions by gurjar112 [here](https://github.com/keras-team/keras/issues/10648). This is a keras 2.2.4 related problem. 

#### Download Physionet Dataset
EEGNet: 
Download the `.edf` files of the Physionet EEG Motor Movement/Imagery Dataset [here](https://physionet.org/content/eegmmidb/1.0.0/), unpack it, and put into the folder `dataset/`

#### Train and Validate Global and Subject-specific Models
Global models are trained and validated in `main_global.py`. Results are in `results/your-global-experiment/stats` and global models in `results/your-experiment-name/model`. 
```
$ (mybciEnv) python3 main_global.py
```

After having trained and stored the global model, they can be refined by doing subject-specific transfer learning (SS-TL) using `main_ss.py`. 
```
$ (mybciEnv) python3 main_ss.py
```
#### Plots
A notebook with selected plots is available under `plot_nbook/`. 


### License and Attribution
Please refer to the LICENSE file for the licensing of our code.
<!--For the pose detection application demo, we heavily modified [this](https://github.com/tensorboy/pytorch_Realtime_Multi-Person_Pose_Estimation) OpenPose implementation. -->

