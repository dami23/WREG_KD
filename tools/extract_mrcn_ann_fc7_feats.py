from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import os.path as osp
import sys
import json
import time
import numpy as np
import h5py
import pprint
from scipy.misc import imread, imresize
import cv2

import torch
from torch.autograd import Variable

# mrcn path
import _init_paths
from mrcn import inference_no_imdb
from loaders.loader import Loader
import pdb

# box functions
def xywh_to_xyxy(boxes):
    """Convert [x y w h] box format to [x1 y1 x2 y2] format."""
    return np.hstack((boxes[:, 0:2], boxes[:, 0:2] + boxes[:, 2:4] - 1))

def image_to_head(head_feats_dir, image_id):
    """Returns
    head: float32 (1, 1024, H, W)
    im_info: float32 [[im_h, im_w, im_scale]]
    """
    feats_h5 = osp.join(head_feats_dir, str(image_id)+'.h5')
    feats = h5py.File(feats_h5, 'r')
    head, im_info = feats['head'], feats['im_info']

    return np.array(head), np.array(im_info)

def ann_to_pool5_fc7(mrcn, image_id, net_conv, im_info):
    """
    Arguments:
        ann: object instance
        net_conv: float32 (1, 1024, H, W)
        im_info: float32 [[im_h, im_w, im_scale]]
    Returns:
        pool5: Variable(cuda) (1, 1024, 7, 7)
        fc7  : Variable(cuda) (1, 2048, 7, 7)
    """
    ann_boxes = xywh_to_xyxy(np.vstack([self.Anns[ann_id]['box'] for ann_id in ann_ids]))
    pool5, fc7 = mrcn.box_to_spatial_fc7(Variable(torch.from_numpy(net_conv).cuda()), im_info, ann_boxes)  
    
    return pool5, fc7

def main(args):
    dataset_splitBy = args.dataset + '_' + args.splitBy
    if not osp.isdir(osp.join('cache/feats/', dataset_splitBy)):
        os.makedirs(osp.join('cache/feats/', dataset_splitBy))

    # Image Directory
    if 'coco' or 'combined' in dataset_splitBy:
        IMAGE_DIR = 'data/images/mscoco/images/train2014'
    elif 'clef' in dataset_splitBy:
        IMAGE_DIR = 'data/images/saiapr_tc-12'
    else:
        print('No image directory prepared for ', args.dataset)
        sys.exit(0)

    # load dataset
    data_json = osp.join('cache/prepro', dataset_splitBy, 'data.json')
    data_h5 = osp.join('cache/prepro', dataset_splitBy, 'data.h5')
    sub_obj_wds = osp.join( 'cache/sub_obj_wds', dataset_splitBy, 'sub_obj_wds.json')
    similarity = osp.join('cache/similarity', dataset_splitBy, 'similarity.json')
    loader = Loader(data_h5=data_h5, data_json=data_json, sub_obj_wds=sub_obj_wds, similarity=similarity)
    
    images = loader.images
    anns = loader.anns
    num_anns = len(anns)
    assert sum([len(image['ann_ids']) for image in images]) == num_anns

    # load mrcn model
    mrcn = inference_no_imdb.Inference(args)

    # feats_h5
    save_path = '/media/mi/Data/Dataset/feats/'
    feats_path = osp.join(save_path, dataset_splitBy, 'mrcn', '%s_%s_%s_ann_fc7' % (args.net_name, args.imdb_name, args.tag))
    if not osp.isdir(feats_path):
        os.makedirs(feats_path)

    # extract
    feats_dir = '%s_%s_%s' % (args.net_name, args.imdb_name, args.tag)
    head_feats_dir = osp.join(save_path, dataset_splitBy, 'mrcn', feats_dir)
    
    for i, image in enumerate(images):
        image_id = image['image_id']
        ann_ids = image['ann_ids']
        
        head, im_info = image_to_head(head_feats_dir, image_id)

        ann_boxes = xywh_to_xyxy(np.vstack([loader.Anns[ann_id]['box'] for ann_id in ann_ids]))
        ann_pool5, ann_fc7 = mrcn.box_to_spatial_fc7(Variable(torch.from_numpy(head).cuda()), im_info, ann_boxes) 

        ann_pool5 = ann_pool5.data.cpu().numpy()
        ann_fc7= ann_fc7.data.cpu().numpy()

        feat_h5 = osp.join(feats_path, str(image['image_id'])+'.h5')
        f = h5py.File(feat_h5, 'w')
        f.create_dataset('ann_pool5', dtype=np.float32, data=ann_pool5)
        f.create_dataset('ann_fc7', dtype=np.float32, data=ann_fc7)
        f.create_dataset('im_info', dtype=np.float32, data=im_info)
        f.close()
            
        if i % 10 == 0:
            print('%s/%s image_id[%s] size[%s] im_scale[%.2f] writen.' % (i+1, len(images), image['image_id'], ann_pool5.shape, im_info[0][2]))
                  

    print('Done.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--imdb_name', default='coco_minus_refer', help='image databased trained on.')
    parser.add_argument('--net_name', default='res101')
    parser.add_argument('--iters', default=1250000, type=int)
    parser.add_argument('--tag', default='notime')

    parser.add_argument('--dataset', type=str, default='refcoco', help='dataset name: refclef, refcoco, refcoco+, refcocog')
    parser.add_argument('--splitBy', type=str, default='unc', help='splitBy: unc, google, berkeley')
    args = parser.parse_args()
    main(args)