*The Original README.md is moved to README_original.md*

-----

##Train ImageNet dataset on faster-rcnn

First, we could download the image files by classes with bounding boxes, so we simply use these data to train our own classifier.

Untar the images and bounding boxes annotations we downloaded from [imagenet](http://imagenet.org) by [wnid] to above tree:
```
!---data
  |---imagenet
    |---Annotations
    |---Images
```

In fact the annotation xml files will less than image files, so we need to use existed annotations to and its corresponded image to train the classifier.

base on the upstream, this project add imagenet dataset get function in to `dataset.factory`, and the drawing box for checking whether the boxes at right place in image.

To draw the boxes on image to new file, run:
```
./lib/datasets/imagenet/draw.py
```
###Train

To train on this dataset, you need to prepare the next steps:


1. change the classes in `./lib/datasets/imagenet/imagenet.py`
2. change the classes num in pt file under `./model/imagenet/{NET}`
3. _fetch pretrained models by run `./data/scripts/fetch_imagenet_models.sh`(optional)_


if your wanna train 'car,dog' classes, so the num will be 3 because of '\_\_background\_\_',change the value in above layers in each files
```
input-data->num_classes = num
class_score->num_output = num
bbox_pred->num_output   = num*4
```
These prototxt was copied from pascal_voc path, so you could copy the prototxt file you need from pascal_voc path for using

then just need to run:
```
./tools/train_faster_rcnn_alt_opt.py --gpu 0\
--net_name ZF \
--weights data/imagenet_models/ZF.v2.caffemodel \
--imdb imagenet \
--cfg experiments/cfgs/faster_rcnn_alt_opt.yml \
--pt_type imagenet
```

In this script, only pt_type is new, the use of others arguments you could see in the demo `./experiments/scripts/faster_rcnn_alt_opt.sh`. pt_type accept `pascal_voc,imagenet` to set the prototxt which will be used.


###Classifier

Change the classes in `./tools/classify`
then run:
```
./tools/classify --net zf
```

so we could get the object detected image like deom.py works.

_Have fun_
