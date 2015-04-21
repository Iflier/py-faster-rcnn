# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

from fast_rcnn_config import cfg, get_output_path
import numpy as np
import cv2
import caffe
import finetuning
import bbox_regression_targets
import os

from caffe.proto import caffe_pb2
import google.protobuf as pb2

class SolverWrapper(object):
    def __init__(self, solver_prototxt, imdb, pretrained_model=None):
        self.bbox_means = None
        self.bbox_stds = None
        self.imdb = imdb

        self.solver = caffe.SGDSolver(solver_prototxt)
        if pretrained_model is not None:
            print 'Loading pretrained model weights from {:s}' \
                .format(pretrained_model)
            self.solver.net.copy_from(pretrained_model)

        self.solver_param = caffe_pb2.SolverParameter()
        with open(solver_prototxt, 'rt') as f:
            pb2.text_format.Merge(f.read(), self.solver_param)

    def snapshot(self):
        if cfg.TRAIN.BBOX_REG:
            assert self.bbox_stds is not None
            assert self.bbox_means is not None

            # save original values
            orig_0 = self.solver.net.params['bbox_pred'][0].data.copy()
            orig_1 = self.solver.net.params['bbox_pred'][1].data.copy()

            # scale and shift with bbox reg unnormalization; then save snapshot
            self.solver.net.params['bbox_pred'][0].data[...] = \
                    (self.solver.net.params['bbox_pred'][0].data *
                     self.bbox_stds[:, np.newaxis])
            self.solver.net.params['bbox_pred'][1].data[...] = \
                    (self.solver.net.params['bbox_pred'][1].data *
                     self.bbox_stds + self.bbox_means)

        output_dir = get_output_path(self.imdb, None)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        infix = ('_' + cfg.TRAIN.SNAPSHOT_INFIX
                 if cfg.TRAIN.SNAPSHOT_INFIX != '' else '')
        filename = self.solver_param.snapshot_prefix + infix + \
                   '_iter_{:d}'.format(self.solver.iter) + '.caffemodel'
        filename = os.path.join(output_dir, filename)

        self.solver.net.save(str(filename))
        print 'Wrote snapshot to: {:s}'.format(filename)

        if cfg.TRAIN.BBOX_REG:
            # restore net to original state
            self.solver.net.params['bbox_pred'][0].data[...] = orig_0
            self.solver.net.params['bbox_pred'][1].data[...] = orig_1

    def train_model(self, roidb, max_iters):
        last_snapshot_iter = -1
        while self.solver.iter < max_iters:
            shuffled_inds = np.random.permutation(np.arange(len(roidb)))
            lim = (len(shuffled_inds) / cfg.TRAIN.IMS_PER_BATCH) * \
                    cfg.TRAIN.IMS_PER_BATCH
            shuffled_inds = shuffled_inds[0:lim]
            for shuffled_i in xrange(0, len(shuffled_inds),
                                     cfg.TRAIN.IMS_PER_BATCH):
                db_inds = shuffled_inds[shuffled_i:shuffled_i +
                            cfg.TRAIN.IMS_PER_BATCH]
                minibatch_db = [roidb[i] for i in db_inds]
                blobs = finetuning.get_minibatch(minibatch_db)

                net = self.solver.net

                for blob_name, blob in blobs.iteritems():
                    # Reshape net's input blobs
                    net.blobs[blob_name].reshape(*(blob.shape))
                    # Copy data into net's input blobs
                    net.blobs[blob_name].data[...] = blob.astype(np.float32,
                                                                 copy=False)

                # Make one SGD update
                self.solver.step(1)

                if self.solver.iter % cfg.TRAIN.SNAPSHOT_ITERS == 0:
                    last_snapshot_iter = self.solver.iter
                    self.snapshot()

                if self.solver.iter >= max_iters:
                    break

        if last_snapshot_iter != self.solver.iter:
            self.snapshot()

def prepare_training_roidb(imdb):
    """
    Enrich the imdb's roidb by adding some derived quantities that
    are useful for training. This function precomputes the maximum
    overlap, taken over ground-truth boxes, between each ROI and
    each ground-truth box. The class with maximum overlap is also
    recorded.
    """
    roidb = imdb.roidb
    for i in xrange(len(imdb.image_index)):
        roidb[i]['image'] = imdb.image_path_at(i)
        # need gt_overlaps as a dense array for argmax
        gt_overlaps = roidb[i]['gt_overlaps'].toarray()
        # max overlap with gt over classes (columns)
        max_overlaps = gt_overlaps.max(axis=1)
        # gt class that had the max overlap
        max_classes = gt_overlaps.argmax(axis=1)
        roidb[i]['max_classes'] = max_classes
        roidb[i]['max_overlaps'] = max_overlaps
        # sanity checks
        # max overlap of 0 => class should be zero (background)
        zero_inds = np.where(max_overlaps == 0)[0]
        assert all(max_classes[zero_inds] == 0)
        # max overlap > 0 => class should not be zero (must be a fg class)
        nonzero_inds = np.where(max_overlaps > 0)[0]
        assert all(max_classes[nonzero_inds] != 0)

    return roidb

def train_net(solver_prototxt, imdb, pretrained_model=None, max_iters=40000):
    # enhance roidb to contain flipped examples
    if cfg.TRAIN.USE_FLIPPED:
        print 'Appending horizontally-flipped training examples...'
        imdb.append_flipped_roidb()
        print 'done'

    # enhance roidb to contain some useful derived quanties
    print 'Preparing training data...'
    roidb = prepare_training_roidb(imdb)
    print 'done'

    # enhance roidb to contain bounding-box regression targets
    print 'Computing bounding-box regression targets...'
    means, stds = \
        bbox_regression_targets.append_bbox_regression_targets(roidb)
    print 'done'

    sw = SolverWrapper(solver_prototxt, imdb, pretrained_model=pretrained_model)
    sw.bbox_means = means
    sw.bbox_stds = stds

    print 'Solving...'
    sw.train_model(roidb, max_iters=max_iters)
    print 'done solving'