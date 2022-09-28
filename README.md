# Segementation-of-colon-cancer-using-unet
- The data is so small, you can download the full data from here https://www.cancerimagingarchive.net/nbia-search/?CollectionCriteria=CT%20COLONOGRAPHY, choose patient 1
- The data is dicom images, so in main.py the data will be converted into png
- After converting the data split into training and validation in the folders
- Draw the mask manually on any annotation tool and save as coco format
- Get the mask images in main.py
- In train.py, pass the images and the mask to the model
- There is weight file unet.h5, you can use it to test
