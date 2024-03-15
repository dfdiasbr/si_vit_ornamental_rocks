# import required libraries
import cv2
import numpy as np
import os

#centering and cropping the image
def center_crop(img, dim):
	"""Returns center cropped image
	Args:
	img: image to be center cropped
	dim: dimensions (width, height) to be cropped
	"""
	width, height = img.shape[1], img.shape[0]

	# process crop width and height for max available dimension
	crop_width = dim[0] if dim[0]<img.shape[1] else img.shape[1]
	crop_height = dim[1] if dim[1]<img.shape[0] else img.shape[0] 
	mid_x, mid_y = int(width/2), int(height/2)
	cw2, ch2 = int(crop_width/2), int(crop_height/2) 
	crop_img = img[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]
	return crop_img

#obtaining the mask from the image
def obtain_mask(image,lower,upper):
    # Convert BGR to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)    
        
    #lower = np.array([0,0,0])
    #upper = np.array([144,57,71])

    lower = np.array(lower)
    upper = np.array(upper)
    
    # Create a mask. Threshold the HSV image to get only colors between lower and upper
    mask = cv2.inRange(hsv, lower, upper)
    
    return mask


#Creating script for
#1 - read the training directory with the images of a class
#2 – resize and rename the images
#3 - create the mask for each image
# 4 - save in \warwick_MIL\CATEGORY\data the class with the resized and renamed images
# 5 - save in \warwick_MIL\CATEGORY\mask the class with the image masks, with the same name as the image


sourcedir = "./sample_datasets/warwick_CLS/"
destdir = "./sample_datasets/warwick_MIL/"
#sourcedir ="../Base de dados do projeto/images_by_category_train_val_test/"
#destdir = "../Base de dados do projeto/processed_images_by_category_train_val_test/"

categories = ['train','val','test']

# classes will be dictionaries with the name and minimum and maximum color values ​​for creating specific masks
# considering analyzes carried out in each class
classes = {
     #'granite-blackswan': (1,[0,0,0],[144,57,71]),
     #'granite-lucyinthesky': (2,[0,6,46],[170,17,202]),
     #'granite-nevascawhite': (3,[0,0,0],[46,39,180]),
     #'marble-dolomite-brancoparana': (4,[0,0,113],[180,44,196]),
     #'marble-dolomite-calacata': (5,[0,0,166],[180,30,250]),
     #'marble-shadow': (6,[0,0,102],[180,245,177]),
     #'quartzite-biancosuperiore': (7,[0,0,129],[180,255,145]),
     'quartzite-oceanblue': (8,[0,0,169],[180,255,195]),
     'quartzite-patagonia': (9,[0,0,166],[180,255,200]),
     #'quartzite-silvermoon': (10,[0,0,153],[180,255,180]),
     #'quartzite-tajmahal': (11,[0,0,144],[180,255,170]),
     #'quartzite-volupia': (12,[0,0,155],[180,255,187])
}

for category in categories:

    directories = []    
    for c in classes.keys():
        directories = np.append(directories,sourcedir+category+'/'+c)
    print(directories)
    
    images = []
    masks = []

    #reduces images to 40% size to apply cropping from the center
    resize_rate = 0.4

    for directory in directories:
        theclass = directory.split('/')[-1]       
        
        if not os.path.exists(destdir+category+'/data/'+theclass):
            os.mkdir(destdir+category+'/data/'+theclass)          
        if not os.path.exists(destdir+category+'/mask/'+theclass):
            os.mkdir(destdir+category+'/mask/'+theclass)
        
        for subdir, dir, files in os.walk(directory):        
            
            #print(subdir)
            #print(destdir+category+'/data/'+theclass)            
                 
            i = 1
            for file in files:
                #print(file)            
                image = cv2.imread(os.path.join(subdir, file))            
                if image is not None:
                    
                    ccrop_img = center_crop(image, (image.shape[1]*resize_rate,image.shape[0]*resize_rate))
                    mask = obtain_mask(ccrop_img,classes[theclass][1],classes[theclass][2])                    
                    
                    cv2.imwrite(destdir+category+'/data/'+theclass+'/img-'+category+'-'+str(i)+'.jpg', ccrop_img)
                    cv2.imwrite(destdir+category+'/mask/'+theclass+'/img-'+category+'-'+str(i)+'.jpg', mask)                    
                i += 1


#for image in images:
    #cv2.imshow('Image',image)
    #cv2.waitKey(0)


# Display cropped image
#cv2.imshow("cropped", ccrop_img)
#cv2.waitKey(0)

# Display mask
#cv2.imshow('Mask',mask)
#cv2.waitKey(0)
#"""    