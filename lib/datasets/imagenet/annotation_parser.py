import os
import xml.dom.minidom

def getText(node):
	return node.firstChild.nodeValue

def getWnid(node):
	return getText(node.getElementsByTagName("name")[0])

def getImageName(node):
	return getText(node.getElementsByTagName("filename")[0])

def getObjects(node):
	objects = []
	for obj in node.getElementsByTagName("object"):
		objects.append({
			"wnid": getText(obj.getElementsByTagName("name")[0]),
			"box":{
				"xmin": int(getText(obj.getElementsByTagName("xmin")[0])),
				"ymin": int(getText(obj.getElementsByTagName("ymin")[0])),
				"xmax": int(getText(obj.getElementsByTagName("xmax")[0])),
				"ymax": int(getText(obj.getElementsByTagName("ymax")[0])),
			}
		})
	return objects

def parse(filepath):
	dom = xml.dom.minidom.parse(filepath)
	root = dom.documentElement
	image_name = getImageName(root)
	wnid = getWnid(root)
	objects = getObjects(root)
	
	return wnid, image_name, objects
