import xml.etree.ElementTree as ET

imgHt = 576 # change according to dataset ht & width 
imgWdt = 960
tree = ET.parse("GT_all_xml1.xml") #give path name of xml file
classes = ['Car', 'Motorcycle', 'Bicycle', 'Human', 'Building', 'Bush', 'Tree', 'Sign_board', 'Road', 'Window']

root = tree.getroot()
print(root[0][1].tag)
print(root[1][0].tag)
for image in root:
	if( image[0].tag == "imageName"):
		imageName = image[0].text
		if imageName[-1] == '0':
			f= open("Data/val.txt","a+")
			f.write("Data/images/"+imageName+".jpg\n" )
		else:
			f= open("Data/train.txt","a+")
			f.write("Data/images/"+imageName+".jpg\n" )
		for child in image[1]:
			if (child.tag == "taggedRectangle"): 
				height = float(child.attrib.get("height"))
				norm_ht = min(height/imgHt, 1.0)
				width = float(child.attrib.get("width"))
				norm_wdt = min(width/imgWdt, 1.0)
				centerX = float(child.attrib.get("x"))+width/2
				norm_centerX = min(centerX/imgWdt, 1.0)
				centerY = float(child.attrib.get("y"))+height/2
				norm_centerY = min(centerY/imgHt, 1.0)
			else :
				if child.text in classes:
					index = str(classes.index(child.text))
					print( index, norm_centerX, norm_centerY, norm_wdt, norm_ht )
					
					f= open("Data/labels/"+imageName+".txt","a+") # path to store the txt file
					f.write(index +" "+ str(norm_centerX) +" "+ str(norm_centerY) +" "+ str(norm_wdt) +" "+  str(norm_ht) +"\n" )



# index centerx/W centery/H w/W h/H
