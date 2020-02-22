# PyTorch Implemation of YOLOv3 to Accomodate Custom Data

This project is a work in progress and issues are welcome.

General Updates/Improvements :
    • User may bring custom data with a custom number of classes 
    • Code cleaner and parameterized 
    • Training has fine-tuning
      
For my Master's thesis I needed to detect objects in a custom dataset of multispectral images. Multispectral images are composed of RGB images, near-infrared images and thermal images. For example, some objects that cannot be visually recognized in the RGB image can be detected in the thermal image. To train our multispectral object detection system, we created a multispectral dataset for object detection. ( Soon, I will add some sample images, detected images & evaluation performance)

Setup :
Install the required Python packages : numpy, torch, torchvision, matplotlib.
Download the full YOLO v3 (237 MB) model in config directory by running "wget https://pjreddie.com/media/files/yolov3.weights" from terminal.

Folder Structure :
All training images and labels are in directory 'Data/images' and 'Data/labels'
Label annotation : one .txt file for each image with same name, different extension (.txt). Each file contains one row for each object, in this format:

class center_x center_y width height

Here, index starting from 0 is used for classes [Car, Motorcycle, Bicycle, Human, Building, Bush, Tree, Sign_board, Road, Window]
center_x, center_y, width & height are normalized by image width and height respectively. So, an example of building bounding box is :
4 0.2171875 0.3324652 0.3364583 0.4218754

The train.txt and val.txt contain the lists of training and validation images, one per line, with full path.
e.g. Data/images/NIR_scene_1.jpg
     Data/images/NIR_scene_2.jpg

Config Files :
Now for the config files in the config/ folder. First, coco.data would look like this:
classes = 10
train=Data/train.txt
valid=Data/val.txt
names=config/coco.names
backup=backup/
I think it’s quite self-explanatory. The backup parameter is not used but seems to be required. The coco.names file is very simple, it’s supposed to list, one per line, the names of the classes (for the annotations file, the first one corresponds to 0, next to 1, etc).

Now, the most important (because if they’re not set correctly, training program will fail) are the classes and final layerfilters values in yolov3.cfg. In the file there are 3 [yolo] sections. Inside that section, set classes to the number of classes in the model. Also have to change the filters value in the [convolutional] section right above [yolo]. That value is equal to:
filters = (classes + 5) x 3
So for my 10 classes, there are 45 filters.

The training program train.py is the standard Yolo script. In the config section, the desired number of epochs,batch size are set, and then run. Depending on the number of training images and hardware, this can take between a couple of hours and more than a day.
The script will save after each epoch. Grab the last file and put it back in your config folder, and then it’s ready to do object detection on custom dataset! Code for the detection functions are in https://github.com/suudii/detect.








