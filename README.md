# MI-BMInet
This repository contains the development of MI-BMInet (previously named edgeEEGNet) from training to quantization and finally to the implementation on Mr. Wolf and Vega (master branch), and the channel selection methods (ch_sel branch).

For more details, please read the paper *MI-BMInet: An Efficient Convolutional Neural Network for Motor Imagery Brain–Machine Interfaces With EEG Channel Selection* available on [arXiv](https://arxiv.org/abs/2203.14592) and on [IEEEXplore](https://ieeexplore.ieee.org/document/10409134). If you find this work useful in your research, please cite
```
@ARTICLE{mibminet,
  author={Wang, Xiaying and Hersche, Michael and Magno, Michele and Benini, Luca},
  journal={IEEE Sensors Journal}, 
  title={MI-BMInet: An Efficient Convolutional Neural Network for Motor Imagery Brain–Machine Interfaces With EEG Channel Selection}, 
  year={2024},
  volume={24},
  number={6},
  pages={8835-8847},
  doi={10.1109/JSEN.2024.3353146}}
```

For an easier usage, you can directly take the model from `MI-BMInet/QuantLab/quantlab/BCI-CompIV-2a/edgeEEGNet/edgeEEGnetbaseline.py` or from `MI-BMInet/QuantLab/quantlab/PhysionetMMMI/edgeEEGNet/edgeEEGnetbaseline.py` (MI-BMInet was previously named edgeEEGNet).


### Acknowledgements
This work received support from Swiss National Science Foundation Project 207913 "TinyTrainer: On-chip Training for TinyML devices"
