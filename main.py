#!/usr/bin/env python3
# pylint: disable=wrong-import-position
# pylint: disable=unused-import

import os
import subprocess
from pathlib import Path
from module import *
from datetime import datetime
import time

dir_name = 'database3.d/'
reference_dir = 'static/'
reference_img = reference_dir+'images_png.d/'
reference_label = reference_dir+'label.d/'
reference_query = reference_dir+'query.d/'

subprocess.run(args='pip3 install -r requirements.txt',
                shell=True,
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT)

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from skimage.io import imread, imshow
from skimage.morphology import erosion, dilation, opening, closing, white_tophat,\
    black_tophat, disk, area_opening
from skimage.color import rgb2gray, rgb2hsv
from skimage.exposure import histogram
from skimage.filters.thresholding import threshold_otsu

from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.preprocessing.image import image_utils
from keras.applications.imagenet_utils import preprocess_input
from keras.models import Model

import cv2

class feature_extraction:
    def __init__(self) -> any: ## when initialize as obj

        base_model = VGG16(weights='imagenet')
        self.model = Model( inputs=base_model.input,\
            outputs=base_model.get_layer('fc2').output )

    def summary(self) -> any:
        """
        ## # Print Summary of the model
        Args:??
        Ret: ??
        Throws: ??
        """
        print( self.model.summary() )

    def extract_pil(self, file_path: str=None):
        """
        ## # Extract Image from obj
        Args: filepath
        ret: ??
        Throws: ??
        """

        # ## # Using CV2 to read image
        # img_cv2 = cv2.imread(file_path)
        # img_cv2 = cv2.resize(img_cv2, (224,224))
        # print(img_cv2.shape[2])
        # img_cv2 = np.array(img_cv2)
        # print(img_cv2[122])
        # print(img_cv2.dtype)
        # img_cv2 = img_cv2.reshape((1, img_cv2.shape[0], img_cv2.shape[1],\
        #                      img_cv2.shape[2]))
        # print(img_cv2.shape)
        # img_cv2 = preprocess_input(img_cv2)
        # feature_cv2 = self.model.predict(img_cv2)[0]
        # print(feature_cv2[0])



        ## # Using PIL to read image
        img = Image.open(file_path)
        print(img.mode)
        img = img.convert('1')
        print(img.mode)
        # img.convert()
        # resize in PIL image obj
        img = img.resize( (224,224) )
        # convert image pixels to image array
        img = np.array(img)
        # img = img * 255
        # print(img.dtype)
        # print(img.shape) # 224x224
        # print(img[122])
        img = np.stack((img,img,img), axis=2)
        # print(img.shape) # 224x224x3
        # print(img[122][122])
        img = np.expand_dims(img, axis=0) # 1x224x224x3

        # img = img.reshape((img.shape[0],img.shape[1],3))

        img = preprocess_input(img)
        feature = self.model.predict(img)[0] # (1, 4096) -> (4096, )
        # print(features[0])
        # features_arr = np.char.mod('%f', features)
        features_vector = feature / np.linalg.norm(feature) # Normalize
        # label = decode_predictions(features_arr)
        # print(label)
    
        # feature = self.model.predict(img)[0]
        # #resize for the model to tensor
        # print(img)
        # print( img.shape[2] )
        # img = img.reshape( (1,img.shape[0],img.shape[1],1) )

        # # Input the image from string
        # image = Image.open(input_image, ) # returns file image
        
        # # Resize
        # image_resize = image.resize( (224, 224) )

        # # Convert Image pixels to image array
        # print ( image_resize ) # image_resize is in PIL.Image.Image image obj
        # image_arr = np.array(image_resize)

        # # resize image for the model
        # image_arr = image_arr.reshape((1,224,224,1))

        # # Prepare the image for VGG model
        # image_arr = preprocess_input(image_arr)
        # print(f'image arr 0: {image_arr.shape[0]}')

        # # x = preprocess_input(x)# extract the features
        # features = self.model.predict(image_arr)[0] # (1, 4096) - > (4096, )
        # # convert from Numpy to a list of values
        # features_arr = np.char.mod('%f', features)
        # print(features_arr)
        return features_vector

    def extract(self, file_path: str=None):
        """
        # Extract feature vector using the VGG16 model
        Args:??
        Ret:??
        Throws:??
        """
        # Read image file using cv2 lib
        img = cv2.imread(filename=file_path)
        # Resize image for VGG16 model 224x224
        img = cv2.resize(img, (224, 224))

        # Reshape image to cater for VGG16 model.
        # print(img)
        img = np.expand_dims(img, axis=0)
        # (1, 224, 224, 3) 1 batch, 224 height, 224 width, 3 color chnnel
        # print( img.shape )

        # Prepare the image for the VGG model
        img = preprocess_input(img)
        features = self.model.predict(img)[0] # (1,4096) -> (4096, )
        print (features)
        
        # Normalize feature vector with Hypothenusss
        features = features/np.linalg.norm(features)
        # np.linalg.norm(features) = sqrt(arr1^2 + arr2^2 ... arrN^2)
        print(features)

        return features
        

    def save(self, input_file, save_target, feature_vector) -> any:
        print(feature_vector.shape)
        np.save(file=f'{save_target}/{input_file.stem}.npy',\
                arr=np.asarray(feature_vector))    

def dataset_label(png_dir: str=reference_img) -> any:
    """
    # ## Invokes a dataset label from ONLY the png directory arugment
    Args: png directory of image to be labelled
    Ret: Label directory path
    Throws: ???
    """
    label_dir = reference_label
    if not os.path.exists(label_dir):
        os.makedirs(name=label_dir)
    
    extractor = feature_extraction()
    extractor.summary()
    for each_input in Path(png_dir).glob('*.png'): # png_dirs='./database.d'
        print(each_input)
        feature_vect = extractor.extract(file_path=str(each_input))
        extractor.save(input_file=each_input, save_target=label_dir, feature_vector=feature_vect)

    return label_dir

def savearr_image(img_arr: np.array) -> any:
    # reference_query = 'reference.d/query.d'
    if not os.path.exists(reference_query):
        os.makedirs(name=reference_query)
    elif os.path.exists(reference_query):
        print(f'Path for refer. query are: {reference_query}')
    
    print(img_arr.dtype)
    img_arr = img_arr.astype('uint8')

    # cv2.imwrite(filename=reference_query, img=img_arr)
    im = Image.fromarray(img_arr)
    im.save(reference_query+'car.png', 'png', optimize=True, quality=100, overwrite=True) # TODO 

def plot_comparison(original: str, filtered: np.ndarray):
    img_path = os.path.join(dir_name, original)
    fig = plt.figure()

    fig.add_subplot(1, 2, 1)
    im = plt.imread(img_path)
    plt.imshow(im)
    
    fig.add_subplot(1, 2, 2)
    plt.imshow(filtered)

    plt.show()

def convertpng(dir_name : os.path='database3.d/') -> any:
    """
    # Convert only gifs image to .png
    # through PIL.Image lib
    Args: 
    dir_name is a string that stores the database of gif images.
    Ret: 0
    Throws: ??
    """
    
    img_ref = reference_img #static/images_png.d/
    if not os.path.exists(img_ref):
        os.makedirs(name=img_ref)
    elif os.path.exists(img_ref):
        print(f'Path for png images are: {img_ref}')

    imgs_name = os.listdir(dir_name)
    for each_image in imgs_name:

        filename, ext = each_image.split('.', 1)[0], each_image.split('.', 1)[1]
        if ext == 'gif': # ~/static/images_png.d/
            img_path = os.path.join(img_ref+filename+'.png')
            each_image = os.path.join( dir_name+each_image )
            img_gif = Image.open(fp=each_image) # each_image=database3.d/octopus-11.gif
            img_gif.save(img_path, 'png', optimize=True, quality=100, overwrite=True)
            print(f'Saved image to: {img_path}\n')
            # cv2.imshow('png', img_gif)
            # cv2.imwrite(img_path, img=img_gif)

        else:
            print(f'File is in format: {ext}')
            break

    # imgs_name = os.listdir(dir_name)
    # for each_image in imgs_name:
    #     filename, ext = each_image.split('.', 1)[0], each_image.split('.', 1)[1]
    #     if ext != 'png':
    #         # print(f'{filename} + {ext}')
    #         img_path = os.path.join(dir_name+each_image) # dir_name=database3.d/ + each_image.png
    #         img_gif = Image.open(img_path) # database3.d/each_image.png
    #         img_path = os.path.join(dir_name+filename)
    #         img_gif.save(img_path+'.png', 'png', optimize=True, quality=100, overwrite=True)
    #         os.remove( dir_name+ filename + '.' + ext ) # Remove old file
    #     elif ext == 'png':
    #         print(f'File is in format: {ext}')
    #         break
    return img_ref

def masked_image(image, mask):
    r = image[:,:,0] * mask
    g = image[:,:,1] * mask
    b = image[:,:,2] * mask
    return np.dstack([r,g,b])

def preprocessing(file_png: str) -> any:
    """
    # Pre-process Image Function using Otsu method
    Assuming the image has a background and foroeground

    Args: file_png, filter, disk_r: int
    Ret: filtered: arr
    Throws:??
    # """
    img = imread(file_png, as_gray=True)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(img.shape)
    imshow(img)
    # plt.show()

    th_value = np.linspace(0, 1, 11) # 0.1, 0.2 ... 1.0
    fig, axis = plt.subplots(2, 5, figsize=(15,8))

    for th, ax in zip(th_value, axis.flatten()):
        img_binarized = img > th
        ax.imshow(img_binarized)
        ax.set_title('$Threshold = %.2f$' % th)

    fig, ax = plt.subplots(1,1,figsize=(12,6))
    th_opt = threshold_otsu(img)
    print(th_opt)
    print(img)
    img_otsu = img > th_opt
    img_otsu = img_otsu * 255
    img_otsu = np.array(img_otsu)
    imshow(img_otsu)
    # print(img_otsu.shape)

    # _filter = masked_image(img, img_otsu)
    # ax[1].imshow(_filter)
    # plt.show()

    if not os.path.exists(reference_query):
        os.makedirs(name=reference_query)
    elif os.path.exists(reference_query):
        print(f'Path for refer. query are: {reference_query}')
    img_arr = img_otsu
    img_arr = np.invert(img_arr)
    img_arr = img_arr.astype('uint8')

    # cv2.imwrite(filename=reference_query, img=img_arr)
    im = Image.fromarray(img_arr)
    im.save(reference_query+f'{str(datetime.today()).split()[0]}_{int(time.time())}.png', 'png', optimize=True, quality=100, overwrite=True) # TODO

    # img = cv2.imread(dir_name+file_png)
    # image = imread(dir_name+file_png)
    # footprint = disk(disk_r)

    # if filter == 'erosion':
    #     filtered = erosion(image, footprint=footprint)

    # elif filter == 'dilation':
    #     filtered = dilation(image, footprint=footprint)

    # elif filter == 'opening':
    #     filtered = opening(image, footprint=footprint)

    # elif filter == 'closing':
    #     filtered = closing(image, footprint=footprint)

    # elif filter == 'white_tophat':
    #     filtered = white_tophat(image, footprint=footprint)

    # elif filter == 'black_tophat':
    #     filtered = black_tophat(image, footprint=footprint)
    im_path = reference_query+f'{str(datetime.today()).split()[0]}_{int(time.time())}.png'

    return th_opt, img_otsu, im_path

def main() -> any:
    """
    # ## Main Entry Point
    Prep the required dataset and vector array for app.py
    """

    # image_jpeg_test = '../audi_PNG1742.png'
    # filtered = preprocessing(file_png=image_jpeg_test, filter='dilation', disk_r=2)
    # plot_comparison(original='ray-9.png', filtered=filtered)


    convertpng(dir_name=dir_name) # returns the png image dir
    dataset_label(reference_img)


    ### 
    extractor_obj = feature_extraction()
    feature_item = [] # storing feature vector
    img_path = [] # storing img path in .png
    img_label = []
    

    # label_dir='/home/japer/divp/reference/labels'
    for each_feature_item in Path(reference_label).glob('*.npy'):
        print(each_feature_item)
        feature_item.append( np.load(each_feature_item, allow_pickle=True) )
        img_path.append( Path('./reference/img/')/(each_feature_item.stem+'.png') )
        img_label.append( (each_feature_item.stem).split('-', 1)[0] ) # split to parts in of '-' ret first part
    feature_vect = np.array(feature_item)

    # /home/japer/divp/reference.d/images_png.d/beetle-1.png
    # output_filename = reference_img+'beetle-1.png' # Check image
    output_filename = '/home/japer/CBIR_system/Yamaha+C40A+Acoustic+Guitar.jpg'
    th_opt, img_arr, im_path = preprocessing(output_filename)
    # print(output_filename)

    query_file = im_path
    query = extractor_obj.extract(file_path=query_file) # ret car
    print (query.shape)
    print (feature_vect.shape)
    dists = np.linalg.norm(feature_vect-query, axis=1)
    print(dists[89])
    ids = np.argsort(dists)
    print(ids)
    euc_dist = [ [dists[id], img_path[id], img_label[id]] for id in ids]
    print(euc_dist[0][2])




if __name__ == '__main__':
        main()