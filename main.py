import numpy as np
import pydicom
from PIL import Image, ImageDraw
import os.path
import json
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator

Path = r"C:\Users\amrye\PycharmProjects\pythonProject2\images"
counter = 0
for i in os.listdir(Path):
    imagePath = Path + "\\" + i
    ds = pydicom.dcmread(imagePath)
    new_image = ds.pixel_array.astype(float)
    scaled_image = (np.maximum(new_image, 0) / new_image.max()) * 255.0
    scaled_image = np.uint8(scaled_image)
    final_image = Image.fromarray(scaled_image)
    final_image.save('image' + str(counter) + '.png')
    counter += 1


# Load json from file
json_file = open("via_project_31Aug2022_9h46m_coco.json")
coco_json = json.load(json_file)
json_file.close()

images = {}
boundlst = []
for i in coco_json["images"]:
    class_name = i['file_name']
    class_id = i['id']
    class_width = i['width']
    class_height = i['height']
    if class_id not in images:
        images[class_id] = []
    images[class_id].append(class_name)
    boundlstin = [int(class_width), int(class_height)]
    images[class_id].append(boundlstin)

annotations = {}
for i in coco_json["annotations"]:
    image_id = i['image_id']
    segmentation = i['segmentation']
    if image_id not in annotations:
        annotations[image_id] = []
    annotations[image_id].append(segmentation)

polygon = []
for i in annotations:
    polygon.append(annotations[i][0][0])

width = []
height = []
for i in images:
    width.append(images[i][1][0])
    height.append(images[i][1][1])

for i in range(10):
    img = Image.new('L', (width[i], height[i]), 0)
    ImageDraw.Draw(img).polygon(polygon[i], outline=1, fill=1)
    mask = np.array(img)
    # plt.imshow(mask)
    # plt.show()
    plt.imsave(f"{i}.png", mask)