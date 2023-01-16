#!/usr/bin/env python3
# pylint: disable=wrong-import-position
# pylint: disable=unused-import

import os
from pathlib import Path


from keras.applications.vgg16 import VGG16
from keras.applications.imagenet_utils import preprocess_input
from keras.models import Model
import cv2
import numpy as np
from PIL import Image

label_dir = './reference/labels'

class dataset_label_obj:
    
    def __init__(self) -> None:
        base_model = VGG16(weights='imagenet')
        self.model = Model(inputs=base_model.input, \
            outputs=base_model.get_layer('fc2').output)

    def extract(self, imgs_path: str=None):
        if not os.path.exists(label_dir):
            os.makedirs(name=label_dir)

        for each_input in Path(imgs_path).glob('*.png'):
            img = Image.open( str(each_input) )
            # resize
            img = img.resize( (224,224) )
            # convert image obj to image arr
            img = np.array(img)

            # resize for the model
            img = np.stack( (img,img,img), axis=2 ) # 224x224x3
            img = np.expand_dims(img, axis=0) # 1x224x224x3

            # prepare image for the VGG model
            img = preprocess_input(img)

            # load image
            feature = self.model.predict(img)[0] # (1, 4096) -> (4096, ?? )
            feature = feature/np.linalg.norm(feature) # Normalize(Hypo)

            # Saving to dir
            np.save(file=f'{label_dir}/{each_input.stem}.npy', arr=np.asarray(feature))


fext = dataset_label_obj()
fext.extract( imgs_path='./database3.d' )