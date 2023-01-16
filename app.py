#!/usr/bin/env python3
# pylint: disable=wrong-input-import-position
# pylint: disable=unused-import

from datetime import datetime
import subprocess
import time
from pathlib import Path
import os

subprocess.run(args='pip3 install -r requirements.txt',
                shell=True,
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT)

from PIL import Image, ImageFile
import numpy as np
from flask import Flask, render_template, url_for, request, redirect

from main import *
from main import feature_extraction as f_extract

dir_name = 'database3.d/'
reference_dir = 'static/'
reference_img = reference_dir+'images_png.d/'
reference_label = reference_dir+'label.d/'
reference_upload = reference_dir+'query.d/'

app = Flask(__name__)

extractor_obj = f_extract()
feature_item = []
img_path = []
img_label = []

for each_item in Path(reference_label).glob('*.npy'):
    feature_item.append( np.load(each_item, allow_pickle=True) )
    img_path.append( Path(reference_img)/(each_item.stem+'.png') )
    img_label.append( (each_item.stem).split('-',1)[0] )
feature_vect = np.array(feature_item)

@app.route('/', methods=['POST', 'GET'])



def index():

    if request.method=='POST':
        print('Hello')
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        file = request.files['query_img']
        no_retrieve = int(request.values['qty_retrieve'])
        img = Image.open(file.stream)
        reference_upload = reference_dir+f'upload.d/{str(datetime.today()).split()[0]}'
        # reference_upload = f'./static/reference.d/upload.d/{str(datetime.today()).split()[0]}'

        if not os.path.exists(reference_upload):
            os.makedirs(reference_upload)
        elif os.path.exists(reference_upload):
            print(f'Folder uploaded image: {reference_upload}')

        output_filename = f'{reference_upload}/{int(time.time())}_{file.filename}'
        img.save(output_filename)
        th_opt, img_arr, im_path = preprocessing(output_filename)
        print(im_path)
        query = extractor_obj.extract(file_path=im_path)
        
        dists = np.linalg.norm(feature_vect-query, axis=1)
        ids = np.argsort(dists)[:no_retrieve]
        euc_distance = [ [dists[id], img_path[id], img_label[id]] for id in ids ]
        print(euc_distance[0][2])

        objectname = (file.filename.split('-',1))[0]



        return render_template('index.html', filename=objectname, query_path=output_filename,\
            scores=euc_distance)
    else:
        return render_template('index.html')

if __name__=='__main__':
    app.run(debug=True)