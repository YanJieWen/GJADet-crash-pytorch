# GJADet-crash-pytorch
Full-Scale Train Collision Multi-Structure Automated Detection

<h2>  
Gradient-guided Joint Representation Loss with Adaptive Neck for Train Crash Detection

</h2>

Central south university

[![Model](https://img.shields.io/badge/GoogleDrive-Weight-blue)](https://drive.google.com/drive/folders/1BI9Iker-qJabpx6B3PaTYAw4pCT8YQJE?usp=drive_link)
</div>

## Contents

- [Abstract](#Abstract)
- [Train](#Train)
- [Test](#Test)
- [Contributing](#contributing)
- [License](#license)

- ## Abstract
  Conducting real train crash experiments is the most straightforward and effective methods to research the train's crashworthiness and enhance passive safety protection. As a non-contact measurement method, high-speed camera  can efficiently capture the evolving motion patterns of trains under the high-speed states. Traditional data extraction methods rely on expert-based manual annotations, which are susceptible to factors such as lighting changes, scale variance, and impact debris. Inspired by the tremendous success of Deep Neural Networks (DNNs) in the computer vision community, we first collect 75 real-world train crash scenes and manually annotated them to form the Crash2024 dataset, enriching the community's data resources. Moreover, we propose the novel Gradient-guided Joint representation loss with Adaptive neck Detection network (GJADet). At the macro level,  we embed the adaptive module into the Path Aggregation Feature Pyramid Network (PAFPN), which combines multiple self-attention mechanisms to achieve scale-awareness, spatial-awareness, and task-awareness approaches, significantly improving the detector's representation ability and alleviating the dense-small characteristics of target points in train crash without significant computational overhead. At the micro level, due to the extreme imbalance of target points compared to other classes, we propose a gradient-guided joint representation classification loss to mitigate the long-tailed distribution problem, and the classification and regression are joint representation to maintain consistency between training and inference. On the Crash2024, our model achieves the performance improvement of 3.5AP and significantly alleviate the accuracy loss problem for rare categories. Our code are open source at [URL](https://github.com/YanJieWen/GJADet-crash-pytorch)


- ## Train
**The overall of framework**

![image](framework.jpg)

We have supplied 7 baseline backbone: [Convnext](https://openaccess.thecvf.com/content/CVPR2022/html/Liu_A_ConvNet_for_the_2020s_CVPR_2022_paper.html), [DLA](https://openaccess.thecvf.com/content_cvpr_2018/html/Yu_Deep_Layer_Aggregation_CVPR_2018_paper.html), [Resnet](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html),[Swin-Transformer](https://openaccess.thecvf.com/content/ICCV2021/html/Liu_Swin_Transformer_Hierarchical_Vision_Transformer_Using_Shifted_Windows_ICCV_2021_paper),[Darknet](https://arxiv.org/abs/1804.02767), [CSPDarknet](https://github.com/ultralytics/ultralytics).  
You can follow the following Tips:
```
change the data path-> line 19
change error path and compare path -> line78, line 79
python test.py
```



