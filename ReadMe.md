###   Defects detection on chips
- Detect mainly three kinds of defects on chips,which are bump,dent and dot at pixel level.
#### Data
- Inputs : 224 * 224 * 6 grey-scale images (3D,2D,pattern,variance,pads,drill hole)
- Outputs : 224 * 224 * 4 binary masks(background,bump,dent,dots)

- Amount : Depends on synthesis settings
#### Models
##### Mask RCNN :  
- Three changes are listed as follows in order to fit chips dataset.
    - Weights of class,bbox,mask loss are 2:1:1. (sparse_categorical_crossentropy,smooth_l1_loss , sparse_softmax_cross_entropy_with_logits)
    - Built a direct path from ROI to final loss. In order to let this model works for null defetcs samples. (Detection target problems and low rate of trainable rois)
    - Add channels of input images and a pretrained subnetworks to merge 6 channels to 3 channels.


- Altogher three modified models are trained and evaluated.
    1.  Backboned with resnet 50. And inputs images withs first 3 channels.
    2.   Backboned with resnet 50. And inputs images with all 6 channels.
    3.   Backbones with resnet 25. And inputs images with first 3 channels.


- Results:
    - Firstly a simple chips lot are trained on model 1 . Details are showed in file *run*,*model_train.ipynb*. After around 20 epochs training on 20K images, f1_score is nearly 97%  , mAP is 0.77. Then a more challenged synthsis chips lot is futhermore be tested. And the best score is 94% by model 1.

Model | Channels | Backbone | F1_Score | Config |
---  |---|---|---|---
Chips_3 | 3 | Resnet 50 | 94%| init with weights trained on simple lot ,12 epoch on all layers
Chips_simple | 3 | Resnet 25 | 87% | randomly init,32 epoch on all layers
Chips_6 | 6 | Resnet 50 | 63% | randomly init, 20 epoch on all layers

- Conclusions:
    1. Pretrained weights is unnecessary ,which is also declared by KAiming He in his recentely paper *Rethinking ImageNet Pre-training*
    2. More channels may disturb the generation of right detection rois. 

##### Unet
- Considering Resnet is a huge network and unperfect performance. Unet with feature pyramid network is explored. Two structures are built on Keras.
  1.  Simple Unet with conv layer and pool layer.
  2.  Merged Resnet 50 with FPN
- Training:
  End to end, input with three channels images and output is binary mask.
- Results:
  1. Both models converged to 94% at only one epoch
  2. Predict results are almost negetive logits.
- Reasons:
  1. Mask for one sample is extremely unbalenced only 20* 20 pixels at most is positive compared to 224 *224 total num.
  2. Training data has around 16% samples which are null defects.
 
##### Conclusion:
Multi task networks performs superior than normal end-end networks. And RPN networks helps model to converge much faster. There are futuremore many other optimizations could be done like cutting images to smaler size like 56*56,try to search hyper parameters of Mask RCNN like positive_targets_rate and training instances numsbers, or merge attention mechanism into main branch.

##### Folders:
- mrcnn: Modified mask rcnn mdoels
- run : Display ,train,eval,files based on mrcnn
- mrcnn6: Mask Rcnn with  6 channels
- run_6 : Train and eval file based on mrcnn6
- Unet : Run main.py with parsesrs


(TBA)


- logs_chips3: Weights of mrcnn model with training data ('/mnt/sh_flex_storage/project/xcos_mask/data/trainset.h5')
- logs_chips4: Weights of mrcnn new generated data based on lot
- logs_chips6: Weights of mrcnn6 model with 6 channels images on new lot
- logs_simply: Weights of simlified mrcnn whose resnet is changed to  25 
