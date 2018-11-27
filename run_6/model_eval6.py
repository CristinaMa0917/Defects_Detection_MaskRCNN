import os
import sys
import numpy as np
import h5py
import warnings
import argparse
import tensorflow as tf
import keras.backend as K
from sklearn.metrics import confusion_matrix as cm
import pandas as pd

# run code default setting
# os.environ['CUDA_VISIBLE_DIVICES']='0'
config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
K.set_session(session)

#argparse
parser = argparse.ArgumentParser()
parser.add_argument('--root_path',default='/mnt/sh_flex_storage/malu/venv/CHIPS_MRCNN',help = 'where the mrcnn model exists')
parser.add_argument('--model_type',default='resnet50')
parser.add_argument('--model_dir',default='/logs_chips3/',help = 'where you choose the model weights ')
parser.add_argument('--model_weights',required=True,help='mask_rcnn_chips_3_0905_null_train.h5' )
parser.add_argument('--test_set',required=True,help='/mnt/sh_flex_storage/malu/venv/multilayer/MaskNet_onehot_True_data3_train.h5')
parser.add_argument('--select_samples',type=int,default = 0,help='how many test samples ')
args = parser.parse_args()

#import mrcnn
ROOT_DIR = os.path.abspath(args.root_path)
sys.path.append(ROOT_DIR)
from mrcnn_6 import utils
import mrcnn_6.model as modellib
from mrcnn_6.chips import ChipsConfig,ChipsDataset


def model_eval(dataset):
    IDs = dataset.image_ids
    if args.select_samples>0:
        image_ids = np.random.choice(IDs,args.select_samples)
    else:
        image_ids = IDs
        
    gt_class = []
    pre_class = []
    ap = []
    score = []
    count = 0
    for image_id in image_ids:
        count += 1
        if count % 500 == 0:
            print('working on %d' % count)
        # Load image and ground truth data
        image, image_meta, gt_class_id, gt_bbox, gt_mask = \
                modellib.load_image_gt(dataset, inference_config,
                                       image_id, use_mini_mask=False)
        results = model.detect([image], verbose=0)
        r = results[0]

        # Compute AP
        if len(r['class_ids']) > 0 and len(gt_class_id) > 0:
            AP, precisions, recalls, overlaps = utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                                                                 r["rois"], r["class_ids"], r["scores"], r['masks'],
                                                                 iou_threshold=0.5)
            # print(AP)
            ap.append(AP)
        else:
            ap.append(0)

        if not len(r['class_ids']):
            pre_class.append(0)
            score.append(0)
        else:
            pre_class.append(r['class_ids'][0])
            score.append(r['scores'][0])

        if not len(gt_class_id):
            gt_class.append(0)
        else:
            gt_class.append(gt_class_id[0])

    # predict summary
    result = pd.DataFrame()
    result['image_id'] = image_ids
    result['gt_class'] = gt_class
    result['pre_class'] = pre_class
    result['ap'] = ap
    result['score'] = score

    return result

def info_display(result):
    image_ids = result['image_id']
    gt_class = result['gt_class']
    pre_class = result['pre_class']
    ap = result['ap']

    # result display
    diff = np.array(gt_class) - np.array(pre_class)
    diff = diff.tolist()
    precision = diff.count(0) / len(diff)
    total_false = len(diff) - diff.count(0)

    negative = len(result[result['gt_class'] == 0])
    positive = len(result) - negative

    false_positive = len(result[result['gt_class'] == 0][result['pre_class'] != 0])
    false_negative = len(result[result['gt_class'] != 0][result['pre_class'] == 0])

    # print('evaluation on %s'%name)
    print('mAP = ', np.mean(ap))
    print('precision =  %d%%' % (precision * 100))
    print('false_positive/total = %d%%' % (false_positive / len(image_ids) * 100))
    print('false_negative/toal = %d%%' % (false_negative / len(image_ids) * 100))

    print('false_positive/negative = %d%%' % ((false_positive / negative) * 100))
    print('false_negative/positive = %d%%' % ((false_negative / positive) * 100))


def error_display(result, num=1):
    y_true = result['gt_class']
    y_pred = result['pre_class']
    cmatrix = cm(y_true, y_pred)

    num_finechips = sum(cmatrix[0])
    num_flawchips = len(result) - num_finechips
    num_pre_finechips = sum(cmatrix[:, 0])
    num_pre_flawchips = len(result) - num_pre_finechips

    print('confusion matrix:\n')
    print(cmatrix)

    pres = []
    recs = []
    for i in range(4):
        precison = cmatrix[i, i] / (sum(cmatrix[i]) +1)* 100
        recall = cmatrix[i, i] / (sum(cmatrix[:, i])+1) * 100
        pres.append(precison)
        recs.append(recall)
        print('precision and recall on class%d ： %d%%   %d%% \n' % (i, precison, recall))

    print('total validation samples num ：  %d' % len(result))
    print('mean presicion ： %d%%' % (np.mean(pres)))
    print('mean recall ： %d%%' % (np.mean(recs)))
    print('mean ap ：  %d%% ' % (np.mean(result['ap']) * 100))
    return

if __name__ == '__main__':
    # prepare data
    test_path = args.test_set
    test_set = h5py.File(test_path, 'r')
    test_images = test_set['input']
    test_masks = test_set['output']
    del test_set
    NUM_TEST = test_images.shape[0]
    dataset_test = ChipsDataset(test_images,test_masks) 
    dataset_test.load_chips(NUM_TEST)
    dataset_test.prepare()

    # code settings
    NUM_TRAIN = 20000
    warnings.filterwarnings('ignore')

    #inference config
    class InferenceConfig(ChipsConfig):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1 # len(images) must be equal to BATCH_SIZE
    inference_config = InferenceConfig()

    # model eval
    MODEL_DIR = args.model_dir
    model_path =os.path.join(ROOT_DIR,MODEL_DIR,args.model_weights)
    if args.model_type == 'resnet50':
        model = modellib.MaskRCNN(mode="inference",
                                  config=inference_config,
                                  model_dir=MODEL_DIR)
    # else:
    #     model = modellib_simply.MaskRCNN(mode="inference",
    #                               config=inference_config,
    #                               model_dir=MODEL_DIR)
    model.load_weights(model_path, by_name=True)
    print('load weights from %s'%model_path)
    
    test_result = model_eval(dataset_test)
    info_display(test_result)
    error_display(test_result)

