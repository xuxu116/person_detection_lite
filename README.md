# Person_detection_lite
Light-weight model with [NCNN](<https://github.com/Tencent/ncnn>) inference for person/pedestrian detection. The model is trained with [maskrcnn](<https://github.com/facebookresearch/maskrcnn-benchmark>) but not for any specified perdestrain detection dataset. You can view the model structure easily with [netron](<https://github.com/lutzroeder/netron>). The pytorch official [mobilenetv2](<https://pytorch.org/hub/pytorch_vision_mobilenet_v2/>) is used as backbone.
It seems the FPN structure and ROIHead produce much computations and will be improved sooner or later. However it is much harder than face detection for tiny networks, expecially for surveillance.

## Model Zoo and Baselines

Flops and params are evaluated by [pytorch-OpCounter](https://github.com/Lyken17/pytorch-OpCounter). More baselines are comming soon..

### Baselines

The mobile0.35xFPNdw and mobile1.0x are not targeting [wider pedestrian](https://competitions.codalab.org/competitions/20132), and no image in wider pedestrain is in the training set. The AP is just for reference.

* mobile1.0x: wider pedestrian AP: 0.377688, params: 2.253M
* mobile0.35xFPNdw: wider pedestrian AP: 0.377688, params: 396.128K

#### MACs vs. input resolution

model | (224, 224) | (480, 288) | (928, 512) | (1152, 640) 
--- | --- | --- | --- | --- 
mobile1.0x | - | 1.334G | 4.585G | 7.115G
mobile1.0x backbone | 326.207M | 898.733M | 3.089G | 4.793G
mobile0.35xFPNdw | - | 508.895M | 1.749G | 2.714G
mobile0.35xFPNdw backbone | 69.457M | 191.361M | 657.714M | 1.021G 


## Demo images

![demo image](images/results/000011_mobile0.35_512_928.jpg)

![demo image](images/results/img00023_mobile0.35_512_928.jpg)



## Usages

- Specified the model path and shape(should be divisible by 32) for input image in the file. Put into the ncnn/examples and build. Or you can directly compile with correctly links. 

- Anchors are built once for the model, which assumes that the input shape remains the same.
