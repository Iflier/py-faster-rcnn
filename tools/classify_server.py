#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
import _init_paths
import os
import json
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
import socket

CLASSES = ('__background__', 'cup', 'glasses', 'tool', 'laptop', 'chair')

NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel'),
        'vgg_cnn_m_1024': ('VGG_CNN_M_1024',
                  'VGG_CNN_M_1024_faster_rcnn_final.caffemodel'),
        'zf': ('ZF',
                  'ZF_faster_rcnn_final.caffemodel')}


def vis_detections(im, class_name, dets, return_boxes, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    boxes = []

    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        boxes.append({"bbox": [int(value) for value in bbox.tolist()], "score": float(score)})
        if not return_boxes:
	    cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 0)
            cv2.putText(im, '{:s} {:.3f}'.format(class_name, score), (bbox[0], bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
    return boxes

def demo(net, im, return_boxes):
    """Detect object classes in an image using pre-computed object proposals."""
    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    classes = {}
    for cls_ind, cls in enumerate(CLASSES[1:]):
        try:
            cls_ind += 1 # because we skipped background
            cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
            keep = nms(dets, NMS_THRESH)
            dets = dets[keep, :]
            bboxes = vis_detections(im, cls, dets, return_boxes, thresh=CONF_THRESH)
            classes[cls] = bboxes
        except Exception as e:
            continue
    if not return_boxes:
        cv2.imshow("image", im)
    return classes

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='vgg16')
    parser.add_argument('--return', dest='return_boxes', default=False,
                        help='whatever return the boxes information')

    args = parser.parse_args()
    args.return_boxes = args.return_boxes == 'True'

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()


    prototxt = os.path.join(cfg.ROOT_DIR, 'models/imagenet', NETS[args.demo_net][0], 'faster_rcnn_alt_opt', 'faster_rcnn_test.pt')
    caffemodel = os.path.join(cfg.ROOT_DIR, 'output/faster_rcnn_alt_opt/imagenet/'+ NETS[args.demo_net][0] +'_faster_rcnn_final.caffemodel')


    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nHave you already run ./tools/train_faster_rcnn_alt_opt.py to train the model?').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(net, im)

    def recvall(sock, count):
        buf = b''
        while count:
            newbuf = sock.recv(count)
            if not newbuf: return None
            buf += newbuf
            count -= len(newbuf)
        return buf

    HOST = ''
    PORT = 6666

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((HOST, PORT))
    s.listen(1)

    while True:
        conn, addr = s.accept()
	if not args.return_boxes:
            cv2.namedWindow("image")
        print("Connected by",addr)

        while True:
            length = recvall(conn,16)

            if not length: break;

            stringData = recvall(conn, int(length))
            image = np.fromstring(stringData, dtype='uint8')
            decoded_image = cv2.imdecode(image, 1)
           
            #scores, boxes = im_detect(net, decoded_image)
            classes = demo(net, decoded_image, args.return_boxes)
            if not args.return_boxes:
                cv2.waitKey(1)
            stringClasses = json.dumps(classes)
            conn.send(str(len(stringClasses)).ljust(16))
            conn.send(stringClasses)

        if not args.return_boxes:
            cv2.destroyWindow("image")
        conn.close()
        print("connection closed")
