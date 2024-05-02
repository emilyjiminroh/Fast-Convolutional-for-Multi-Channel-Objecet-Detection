# Fast-Convolutional-for-Multi-Channel-Objecet-Detection
Fast convolutional for multi-channel object detection and evaluation inspired by SFA3D

SFA3D:
" Super Fast and Accurate 3D Object Detection based on 3D LiDAR Point Clouds"
'''
@misc{Super-Fast-Accurate-3D-Object-Detection-PyTorch,
  author =       {Nguyen Mau Dung},
  title =        {{Super-Fast-Accurate-3D-Object-Detection-PyTorch}},
  howpublished = {\url{https://github.com/maudzung/Super-Fast-Accurate-3D-Object-Detection}},
  year =         {2020}
}
'''

## 1.  Hierarchy
```
${ROOT}
└── checkpoints/
    ├── fpn_resnet_18/    
        ├── fpn_resnet_18_epoch_300.pth
└── dataset/    
    └── kitti/
        ├──ImageSets/
        │   ├── test.txt
        │   ├── train.txt
        │   └── val.txt
        ├── training/
        │   ├── image_2/ (left color camera)
        │   ├── calib/
        │   ├── label_2/
        │   └── velodyne/
        └── testing/  
        │   ├── image_2/ (left color camera)
        │   ├── calib/
        │   └── velodyne/
        └── classes_names.txt
└── sfa/
    ├── config/
    │   ├── train_config.py
    │   └── kitti_config.py
    ├── data_process/
    │   ├── kitti_dataloader.py
    │   ├── demo_dataset.py
    │   ├── kitti_bev_utils.py
    │   ├── kitti_dataset.py
    │   ├── transformation.py
    │   └── kitti_data_utils.py
    ├── losses/
    │   └── losses.py
    ├── models/
    │   ├── fpn_resnet.py
    │   ├── resnet.py
    │   └── model_utils.py
    └── utils/
    │   ├── box_np_ops.py
    │   ├── classic_utils.py
    │   ├── demo_utils.py
    │   ├── eval.py
    │   ├── evaluate.py
    │   ├── evaluation_utils.py
    │   ├── kitti_common.py
    │   ├── logger.py
    │   ├── lr_scheduler.py
    │   ├── misc.py
    │   ├── nms_gpu.py
    │   ├── rotate_iou.py
    │   ├── nms_gpu.py
    │   ├── rotate_iou.py
    │   ├── torch_utils.py
    │   ├── train_utils.py
    │   └── visualization_utils.py
    ├── classical_demo.ipynb
    ├── classical_train_eval.ipynb
    ├── demo_2_sides.py
    ├── demo_front.py
    ├── test.py
    └── train.py
├── README.md 
└── requirements.txt
```


## 2. DATASET
3D KITTI Dataset은[Link](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d).
에서 다운 받을 수 있습니다.
구성 요소는 다음과 같습니다:

- Velodyne point clouds _**(29 GB)**_
- Training labels of object data set _**(5 MB)**_
- Camera calibration matrices of object data set _**(16 MB)**_
- **Left color images** of object data set _**(12 GB)**_ (For visualization purpose only)

## 3. Train
### Train Code
'''
!python train.py --gpu_idx 0 --mode classic #--mode checkpoints folder
'''

## 4. Evaluate (AP)
### Evaluate Code
'''
!python train.py --evaluate  --gpu_idx 0 --resume_path './model_path'
'''


## 5. Visualization (Demo)
### Demo Code
'''
!python demo_2_sides.py --gpu_idx 0 --peak_thresh 0.2 --saved_fn final_demo_classic
'''


 ## REFERENCE
[1] "Object Detector for Autonomous Vehicles Based on Improved Faster RCNN": [2D 이전 버전](https://github.com/Ziruiwang409/improved-faster-rcnn/blob/main/README.md) <br/>
[2]"Torch-quantum"[QNN Implementation](https://github.com/mit-han-lab/torchquantum) <br/>
[3] "KITTI-WAYMO Adapter": [WAYMO데이터 활용](https://github.com/JuliaChae/Waymo-Kitti-Adapter) <br/>
[4] CenterNet: [Objects as Points paper](https://arxiv.org/abs/1904.07850), [PyTorch Implementation](https://github.com/xingyizhou/CenterNet) <br>
[5] RTM3D: [PyTorch Implementation](https://github.com/maudzung/RTM3D) <br>
[6] Libra_R-CNN: [PyTorch Implementation](https://github.com/OceanPang/Libra_R-CNN)

_The YOLO-based models with the same BEV maps input:_ <br>
[7] Complex-YOLO: [v4](https://github.com/maudzung/Complex-YOLOv4-Pytorch), [v3](https://github.com/ghimiredhikura/Complex-YOLOv3), [v2](https://github.com/AI-liu/Complex-YOLO)

*3D LiDAR Point pre-processing:* <br>
[8] VoxelNet: [PyTorch Implementation](https://github.com/skyhehe123/VoxelNet-pytorch)

