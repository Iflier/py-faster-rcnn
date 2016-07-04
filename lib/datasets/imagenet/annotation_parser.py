import os
import xml.dom.minidom

def getText(node):
	return node.firstChild.nodeValue

def getClassId(node):
	return getText(node.getElementsByTagName("name")[0])

def getImageName(node):
	return getText(node.getElementsByTagName("filename")[0])

def getBoxs(node):
	boxs = []
	for v in node.getElementsByTagName("bndbox"):
		boxs.append({
			"xmin": int(getText(v.getElementsByTagName("xmin")[0])),
			"ymin": int(getText(v.getElementsByTagName("ymin")[0])),
			"xmax": int(getText(v.getElementsByTagName("xmax")[0])),
			"ymax": int(getText(v.getElementsByTagName("ymax")[0])),
		})
	return boxs

def parse(filepath):
	dom = xml.dom.minidom.parse(filepath)
	root = dom.documentElement
	image_name = getImageName(root)
	class_id = getClassId(root)
	boxs = getBoxs(root)
	
	return class_id, image_name, boxs
