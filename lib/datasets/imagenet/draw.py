#!/usr/bin/env python
import cv2
import os
import annotation_parser as ap

path = "data/imagenet"
annotations_path = os.path.join(path, "Annotations")
images_path = os.path.join(path, "Images")
images_withbox_path = os.path.join(path, "Images_withbox")

if not os.path.exists(images_withbox_path):
	os.mkdir(images_withbox_path)

for wnid_xml in list(os.walk(annotations_path))[0][2]:
	wnid, image_name, objects = ap.parse(os.path.join(annotations_path, wnid_xml))
	image_full_name = image_name + ".JPEG"	

	img = cv2.imread(os.path.join(images_path, image_full_name))
	#if img is None: continue
	for obj in objects:
		box = obj["box"]
		cv2.rectangle(img, (box["xmin"], box["ymin"]), (box["xmax"], box["ymax"]), (0, 255, 0), 1, 0)
	cv2.imwrite(os.path.join(images_withbox_path, image_full_name), img)
	print("saved " + image_full_name)
