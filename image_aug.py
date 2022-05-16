import albumentations as A
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys
from tqdm import tqdm
import os
import time

def visualize(image, bboxes):
    fig, ax = plt.subplots()
    ax.imshow(image)
    for box in bboxes:
        rect = patches.Rectangle((box[0]*3024 - box[2]*3024/2, box[1]*4032 - box[3]*4032/2), box[2]*3024, box[3]*4032, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    plt.show()

base = '/home/kfkoe2/Research/Pests/Japanese_Beetle_samples/'
file_list = []
for r, d, f in os.walk(base + 'Original'):
    for file in f:
        if '.jpg' in file:
            file_list.append(os.path.join(r, file))
            
file_list = sorted(file_list, key=lambda x: int(os.path.splitext(x)[0][-1]))
transforms_per_image = 10
with tqdm(range(len(file_list) * transforms_per_image), desc="Generating Images") as pbar:
    for file in file_list:
        
        name = file[file.rindex('/') + 1: -4] # Name of jpg without extension
        print("Augmenting: ", name)
        count = 0
        while count < transforms_per_image:
            # Declare an augmentation pipeline
            transform = A.Compose([
                A.RandomResizedCrop(width=3024, height=4032),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.Rotate(p=0.5)
            ], bbox_params=A.BboxParams(format='yolo', min_visibility=.25, label_fields=['class_labels']))

            

            # Read an image with OpenCV and convert it to the RGB colorspace
            image = cv2.imread(f'{base}/Original/{name}.jpg')
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            
            # Get class labels
            class_labels = []

            bboxes = []
            with open(f'{base}/Original/{name}.txt') as f:
                lines = f.readlines()

            for line in lines:
                bboxes.append(list(map(float,line.split()))[1:])
                class_labels.append("Japanese Beetle")

            #print(bboxes)
            #visualize(image, bboxes)

            #sys.exit()
            # Transform Images
            transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
            transformed_image = transformed["image"]
            transformed_image = cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB)
            transformed_bboxes = transformed["bboxes"]
            transformed_class_labels = transformed["class_labels"]
            #visualize(transformed_image, transformed_bboxes)
            # Check if augmentations have bounding boxes
            if(len(transformed_bboxes) > 0):
                cv2.imwrite(f'{base}/Augmented/{name}_{count}.jpg', transformed_image)
                with open(f'{base}/Augmented/{name}_{count}.txt', 'w') as f:
                    for box in transformed_bboxes:
                        f.write("0 ")
                        for entry in box:
                            f.write(str(entry) + " ")
                        f.write('\n')
                count += 1
                pbar.update(1)
        #sys.exit()    
