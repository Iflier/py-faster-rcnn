# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import os
from datasets.imdb import imdb
import datasets.ds_utils as ds_utils
import xml.etree.ElementTree as ET
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
import cPickle
import subprocess
import uuid
from voc_eval import voc_eval
import annotation_parser as ap
from fast_rcnn.config import cfg

class imagenet(imdb):
    def __init__(self, image_set):
        imdb.__init__(self, 'imagenet')
        self._image_set = image_set
        self._data_path = os.path.join(cfg.DATA_DIR, "imagenet")
        
        self._class_wnids = {
            'cup': 'n03147509',
            'glasses': 'n04272054'
        }
        self._classes = ('__background__', self._class_wnids['cup'], self._class_wnids['glasses'])
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        print(self._class_to_ind)
        self._xml_path = os.path.join(self._data_path, "Annotations")
        self._image_ext = '.JPEG'
	# the xml file name and each one corresponding to image file name
        self._image_index = self._load_xml_filenames()
        self._salt = str(uuid.uuid4())
        self._comp_id = 'comp4'

        # PASCAL specific config options
        self.config = {'cleanup'     : True,
                       'use_salt'    : True,
                       'use_diff'    : False,
                       'matlab_eval' : False,
                       'rpn_file'    : None,
                       'min_size'    : 2}

        assert os.path.exists(self._data_path), \
                'Path does not exist: {}'.format(self._data_path)


    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_image_filename(self._image_index[i])

    def image_path_from_image_filename(self, image_filename):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._data_path, 'Images',
                                  image_filename + self._image_ext)
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_xml_filenames(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        xml_folder_path = os.path.join(self._data_path, "Annotations")
        assert os.path.exists(xml_folder_path), \
            'Path does not exist: {}'.format(xml_folder_path)

	for dirpath, dirnames, filenames in os.walk(xml_folder_path):
		xml_filenames = [xml_filename.split(".")[0] for xml_filename in filenames]

        return xml_filenames

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = [self._load_imagenet_annotation(xml_filename)
                    for xml_filename in self._image_index]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb

    def rpn_roidb(self):
        if self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            rpn_roidb = self._load_rpn_roidb(gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, rpn_roidb)
        else:
            roidb = self._load_rpn_roidb(None)

        return roidb

    def _load_rpn_roidb(self, gt_roidb):
        filename = self.config['rpn_file']
        print 'loading {}'.format(filename)
        assert os.path.exists(filename), \
               'rpn data not found at: {}'.format(filename)
        with open(filename, 'rb') as f:
            box_list = cPickle.load(f)
        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_imagenet_annotation(self, xml_filename):
        """
        Load image and bounding boxes info from XML file in the ImageNet format
        """
        filepath = os.path.join(self._data_path, 'Annotations', xml_filename + '.xml')
        wnid, image_name, objects = ap.parse(filepath)
        num_objs = len(objects)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        seg_areas = np.zeros((num_objs), dtype=np.float32)

        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objects):
            box = obj["box"]
            x1 = box['xmin']
            y1 = box['ymin']
            x2 = box['xmax']
            y2 = box['ymax']
            # go next if the wnid not exist in declared classes
            try:
                cls = self._class_to_ind[obj["wnid"]]
            except KeyError:
		print "wnid %s isn't show in given"%obj["wnid"]
                continue
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'flipped' : False,
                'seg_areas' : seg_areas}

    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True

if __name__ == '__main__':
    from datasets.imagenet import imagenet
    d = pascal_voc('train')
    res = d.roidb
    from IPython import embed; embed()
