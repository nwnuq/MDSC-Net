
import os
import random
import numpy as np
from skimage import io
from PIL import Image

from tqdm import tqdm
root_dir = 'I:\\Lab Files\\data\\ISIC 2018\\Task 1'                # change it in your saved original data path
save_dir = 'I:\\Lab Files\\data\\ISIC 2018\\Task 1\\preprocess'


if __name__ == '__main__':
    imgfile = os.path.join(root_dir, 'ISIC2018_Task1-2_Training_Input')
    labfile = os.path.join(root_dir, 'ISIC2018_Task1_Training_GroundTruth')
    filename = sorted([os.path.join(imgfile, x) for x in os.listdir(imgfile) if x.endswith('.jpg')])
    random.shuffle(filename)
    labname = [filename[x].replace('ISIC2018_Task1-2_Training_Input', 'ISIC2018_Task1_Training_GroundTruth'
                                   ).replace('.jpg', '_segmentation.png') for x in range(len(filename))]

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir+'\\image')
        os.makedirs(save_dir+'\\label')

    for i in tqdm(range(len(filename))):
        fname = filename[i].rsplit('\\', maxsplit=1)[-1].split('.')[0]
        lname = labname[i].rsplit('\\', maxsplit=1)[-1].split('.')[0]

        image = Image.open(filename[i])
        label = Image.open(labname[i])

        image = image.resize((342, 256))
        label = label.resize((342, 256))
        image = np.array(image)
        label = np.array(label)
        images_img_filename = os.path.join(save_dir, 'image', fname)
        labels_img_filename = os.path.join(save_dir, 'label', lname)
        np.save(images_img_filename, image)
        np.save(labels_img_filename, label)
    print('Successfully saved preprocessed data')
