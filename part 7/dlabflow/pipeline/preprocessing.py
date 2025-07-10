from functools import partial
from kfp.components import create_component_from_func
import kfp
from kfp import onprem
from kfp import compiler
from kfp import dsl
from kfp.dsl import component
import argparse
import requests
import asyncio
import bentoml
from pydantic import BaseModel, ValidationError 
import pydantic
import typing as t
from typing import Any, Type
import requests
import json
import re

################################################################################################
## kubeflow pipeline
################################################################################################

@partial(create_component_from_func, base_image='dgkim1983/dlabflow:yolo-24061401', packages_to_install=['minio', 'split-folders', 'pymysql'])
def Preprocessing(projectId: str, versionId: str, dataPath: str, dataNormalization: str, dataAugmentation: str, trainRatio: int, validationRatio: int, testRatio: int):
    import os
    import xml.etree.ElementTree as ET
    import cv2
    import random
    import matplotlib.pyplot as plt
    import pandas as pd
    from distutils.dir_util import copy_tree
    import splitfolders
    import shutil
    from minio import Minio
    import csv
    from pathlib import Path
    import pymysql
    import sys

    ################################################################################################
    ## data path
    ################################################################################################

    bucket = 'aiproject'
    dataPath = ''

    base_path = '/mnt/dlabflow/backend/minio/'+bucket

    tmp_annotation_path = base_path+'/'+projectId+'/rawdata/annotations/'
    tmp_image_path = base_path+'/'+projectId+'/rawdata/images/'

    client = Minio(endpoint='10.40.217.236:9002', access_key='dlab-backend', secret_key='dlab-backend-secret', secure=False)
    for i in os.listdir(tmp_annotation_path):
        client.fput_object(bucket_name=bucket, object_name='/'+projectId+'/rawdata/annotations/'+i, file_path=tmp_annotation_path+'/'+i)
    for i in os.listdir(tmp_image_path):
        client.fput_object(bucket_name=bucket, object_name='/'+projectId+'/rawdata/images/'+i, file_path=tmp_image_path+'/'+i)

    minio_path = '/mnt/dlabflow/backend/minio/'+bucket
    annotation_path = tmp_annotation_path
    annotation_list = sorted([f for f in os.listdir(annotation_path) if f.endswith(tuple(['xml', 'txt', 'csv']))])
    result_path = minio_path+'/'+projectId+'/'+versionId+'/preprocessing'
    tmps = result_path+'/tmp/a'
    tmps_split = result_path+'/tmpsplit'

    ################################################################################################
    ## visualization
    ################################################################################################

    def sample(data, color, value):
        image_list = sorted([f for f in os.listdir(data) if f.endswith(tuple(['jpg', 'png', 'JPG', 'PNG']))])
        files = []
        if len(image_list) == 10:
            for i in range(len(random.sample(image_list, 10))):
                f = data+'/'+image_list[i]
                files.append(f)
        else:
            for i in range(len(random.sample(image_list, len(image_list)))):
                f = data+'/'+image_list[i]
                files.append(f)
        fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(11.69, 8.27), tight_layout=True)
        axes = axes.flatten()
        for idx, (ax, file) in enumerate(zip(axes, files)):
            pic = plt.imread(file)
            if color == 'gray':
                ax.imshow(pic, cmap=plt.get_cmap('gray'))
                ax.axis('off')
            else:
                ax.imshow(pic)
                ax.axis('off')
        else:
            [ax.set_visible(False) for ax in axes.flatten()[idx+1:]]
        os.makedirs(result_path+'/sample/', exist_ok=True)
        fig.savefig(result_path+'/sample/'+value+'_sample.jpg', dpi=300)
        for i in os.listdir(result_path+'/sample'):
            client.fput_object(bucket_name=bucket, object_name=projectId+'/'+versionId+'/preprocessing/sample/'+i, file_path=result_path+'/sample/'+i)

    ################################################################################################
    ## preprocessing task
    ################################################################################################

    def preprocessings():
        def dataframe():
            data_dict = {'filename':[], 'label':[], 'xmin': [], 'ymin': [], 'xmax': [], 'ymax': []}
            for i in annotation_list:
                tree = ET.parse(annotation_path+i)
                root = tree.getroot()
                filename = root.find('filename').text
                for obj in root.findall('object'):
                    label = obj.find('name').text
                    bndbox_tree = obj.find('bndbox')
                data_dict['filename'].append(filename)
                data_dict['label'].append(label)
                data_dict['xmin'].append(bndbox_tree.find('xmin').text)
                data_dict['ymin'].append(bndbox_tree.find('ymin').text)
                data_dict['xmax'].append(bndbox_tree.find('xmax').text)
                data_dict['ymax'].append(bndbox_tree.find('ymax').text)
            df_data = pd.DataFrame(data_dict)
            unique = pd.unique(df_data['filename'])
            color = (0, 255, 0)
            thickness = 2
            return df_data, unique, color, thickness

        ############################################################################################
        ## preprocessing task 1 : grayscale
        ############################################################################################

        def grayscale():
            os.makedirs(result_path+'/normalization/'+normalization_type, exist_ok=True)
            image_path = tmp_image_path
            image_list = sorted([f for f in os.listdir(image_path) if f.endswith(tuple(['jpg', 'png', 'JPG', 'PNG']))])
            for i, j in zip(image_list, annotation_list):
                image = cv2.imread(image_path+i, cv2.IMREAD_COLOR)
                image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                tree = ET.parse(annotation_path+j)
                root = tree.getroot()
                folder = root.find('folder').text
                filename = root.find('filename').text
                name, ext = os.path.splitext(filename)
                new_filename = f"{name}_{normalization_type}{ext.lower()}"
                cv2.imwrite(result_path+'/normalization/'+normalization_type+'/'+new_filename, image_gray)
                root.find('filename').text = str(new_filename)
                root.find('path').text = str(folder+'/'+new_filename)
                tree.write(result_path+'/normalization/'+normalization_type+'/'+j.rstrip('.xml')+'_'+normalization_type+'.xml')
            for i in os.listdir(result_path+'/normalization/'+normalization_type):
                client.fput_object(bucket_name=bucket, object_name=projectId+'/'+versionId+'/preprocessing/normalization/'+normalization_type+'/'+i, file_path=result_path+'/normalization/'+normalization_type+'/'+i)

        ############################################################################################
        ## preprocessing task 2 : reverse left-right, reverse top-bottom
        ############################################################################################

        def reverse(image_path):
            os.makedirs(result_path+'/augmentation/'+augmentation_type, exist_ok=True)
            image_list = sorted([f for f in os.listdir(image_path) if f.endswith(tuple(['jpg', 'png', 'JPG', 'PNG']))])
            def flip_bounding_box(reverse_select, bbox, image_width, image_height):
                # 1은 좌우 반전, 0은 상하 반전
                if reverse_select == 0:
                    x_min, y_min, x_max, y_max = bbox
                    new_y_min = image_height - y_max
                    new_y_max = image_height - y_min
                    return x_min, new_y_min, x_max, new_y_max
                else:
                    x_min, y_min, x_max, y_max = bbox
                    new_x_min = image_width - x_max
                    new_x_max = image_width - x_min
                    return new_x_min, y_min, new_x_max, y_max
            df_data, unique, color, thickness = dataframe()
            for i, count in zip(image_list, range(len(unique))):
                image_nn = cv2.imread(image_path+i, cv2.IMREAD_COLOR)
                flipped_image = cv2.flip(image_nn, reverse_select)
                for j in range(0, len(df_data)):
                    if df_data.iloc[j].filename == unique[count]:
                        bounding_box = [int(df_data['xmin'][j]), int(df_data['ymin'][j]), int(df_data['xmax'][j]), int(df_data['ymax'][j])]
                        flipped_bbox = flip_bounding_box(reverse_select, bounding_box, image_nn.shape[1], image_nn.shape[0])
                        #cv2.rectangle(flipped_image, (flipped_bbox[0], flipped_bbox[1]), (flipped_bbox[2], flipped_bbox[3]), color, thickness)
                cv2.imwrite(result_path+'/augmentation/'+augmentation_type+'/'+os.path.splitext(i)[0]+'_'+augmentation_type+'.jpg', flipped_image)
            for i, j in zip(image_list, annotation_list):
                image = cv2.imread(image_path+i)
                tree = ET.parse(annotation_path+j)
                root = tree.getroot()
                filename = root.find('filename').text
                root.find('filename').text = str(os.path.splitext(i)[0]+'_'+augmentation_type+'.jpg')
                root.find('path').text = str(os.path.splitext(i)[0]+'_'+augmentation_type+'.jpg')
                for obj in root.findall('object'):
                    label = obj.find('name').text
                    bndbox_tree = obj.find('bndbox')
                    bounding_box = [int(bndbox_tree.find('xmin').text), int(bndbox_tree.find('ymin').text), int(bndbox_tree.find('xmax').text), int(bndbox_tree.find('ymax').text)]
                    flipped_bbox = flip_bounding_box(reverse_select, bounding_box, image.shape[1], image.shape[0])
                    bndbox_tree.find('xmin').text = str(flipped_bbox[0])
                    bndbox_tree.find('ymin').text = str(flipped_bbox[1])
                    bndbox_tree.find('xmax').text = str(flipped_bbox[2])
                    bndbox_tree.find('ymax').text = str(flipped_bbox[3])
                tree.write(result_path+'/augmentation/'+augmentation_type+'/'+os.path.splitext(i)[0]+'_'+augmentation_type+'.xml')
            for i in os.listdir(result_path+'/augmentation/'+augmentation_type):
                client.fput_object(bucket_name=bucket, object_name=projectId+'/'+versionId+'/preprocessing/augmentation/'+augmentation_type+'/'+i, file_path=result_path+'/augmentation/'+augmentation_type+'/'+i)

        ############################################################################################
        ## preprocessing task 3 : rotation 90 & 180 & 270
        ############################################################################################

        def rotation(image_path):
            os.makedirs(result_path+'/augmentation/'+augmentation_type, exist_ok=True)
            image_list = sorted([f for f in os.listdir(image_path) if f.endswith(tuple(['jpg', 'png', 'JPG', 'PNG']))])
            def rotate_bounding_box(bbox, image_width, image_height, types):
                x_min, y_min, x_max, y_max = bbox
                if types == 'rotation_90':
                    new_x_min = image_height - y_max
                    new_y_min = x_min
                    new_x_max = image_height - y_min
                    new_y_max = x_max
                elif types == 'rotation_180':
                    new_x_min = image_width - x_max
                    new_y_min = image_height - y_max
                    new_x_max = image_width - x_min
                    new_y_max = image_height - y_min
                elif types == 'rotation_270':
                    new_x_min = y_min
                    new_y_min = image_width - x_max
                    new_x_max = y_max
                    new_y_max = image_width - x_min
                return new_x_min, new_y_min, new_x_max, new_y_max
            df_data, unique, color, thickness = dataframe()
            for i, count in zip(image_list, range(len(unique))):
                image = cv2.imread(image_path+i, cv2.IMREAD_COLOR)
                if augmentation_type == 'rotation_90':
                    image_rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                elif augmentation_type == 'rotation_180':
                    image_rotated = cv2.rotate(image, cv2.ROTATE_180)
                elif augmentation_type == 'rotation_270':
                    image_rotated = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                image = cv2.cvtColor(image_rotated, cv2.COLOR_BGR2RGB)
                plt.axis('off')
                for j in range(0, len(df_data)):
                    if df_data.iloc[j].filename == unique[count]:
                        bounding_box = [int(df_data['xmin'][j]), int(df_data['ymin'][j]), int(df_data['xmax'][j]), int(df_data['ymax'][j])]
                        if augmentation_type == 'rotation_90':
                            rotated_bbox = rotate_bounding_box(bounding_box, image.shape[1], image.shape[0], augmentation_type)
                        elif augmentation_type == 'rotation_180':
                            rotated_bbox = rotate_bounding_box(bounding_box, image.shape[1], image.shape[0], augmentation_type)
                        elif augmentation_type == 'rotation_270':
                            rotated_bbox = rotate_bounding_box(bounding_box, image.shape[1], image.shape[0], augmentation_type)
                        #cv2.rectangle(image, (rotated_bbox[0], rotated_bbox[1]), (rotated_bbox[2], rotated_bbox[3]), color, thickness)
                cv2.imwrite(result_path+'/augmentation/'+augmentation_type+'/'+os.path.splitext(i)[0]+'_'+augmentation_type+'.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            for i, j in zip(image_list, annotation_list):
                image = cv2.imread(image_path+i)
                tree = ET.parse(annotation_path+j)
                root = tree.getroot()
                filename = root.find('filename').text
                root.find('filename').text = str(os.path.splitext(i)[0]+'_'+augmentation_type+'.jpg')
                root.find('path').text = str(os.path.splitext(i)[0]+'_'+augmentation_type+'.jpg')
                for obj in root.findall('object'):
                    label = obj.find('name').text
                    bndbox_tree = obj.find('bndbox')
                    bounding_box = [int(bndbox_tree.find('xmin').text), int(bndbox_tree.find('ymin').text), int(bndbox_tree.find('xmax').text), int(bndbox_tree.find('ymax').text)]
                    if augmentation_type == 'rotation_90':
                        rotated_bbox = rotate_bounding_box(bounding_box, image.shape[1], image.shape[0], augmentation_type)
                    elif augmentation_type == 'rotation_180':
                        rotated_bbox = rotate_bounding_box(bounding_box, image.shape[1], image.shape[0], augmentation_type)
                    elif augmentation_type == 'rotation_270':
                        rotated_bbox = rotate_bounding_box(bounding_box, image.shape[1], image.shape[0], augmentation_type)
                    bndbox_tree.find('xmin').text = str(rotated_bbox[0])
                    bndbox_tree.find('ymin').text = str(rotated_bbox[1])
                    bndbox_tree.find('xmax').text = str(rotated_bbox[2])
                    bndbox_tree.find('ymax').text = str(rotated_bbox[3])
                tree.write(result_path+'/augmentation/'+augmentation_type+'/'+os.path.splitext(i)[0]+'_'+augmentation_type+'.xml')
            for i in os.listdir(result_path+'/augmentation/'+augmentation_type):
                client.fput_object(bucket_name=bucket, object_name=projectId+'/'+versionId+'/preprocessing/augmentation/'+augmentation_type+'/'+i, file_path=result_path+'/augmentation/'+augmentation_type+'/'+i)

        ############################################################################################
        ## preprocessing task 4 : train, validation, test dataset split
        ############################################################################################

        check_dataAugmentation = dataAugmentation.replace(",", "").split()

        check_dataAugmentation = ['reverse_tb' if x == 'rotation_TB' else 'reverse_lr' if x == 'rotation_LR' else x for x in check_dataAugmentation]
        print(check_dataAugmentation)

        if 'grayscale' in dataNormalization:
            normalization_type = 'grayscale'
            grayscale()
            sample(result_path+'/normalization/'+normalization_type, 'gray', normalization_type)

            # select = 1
            if len(check_dataAugmentation) == 1:
                if all(x in check_dataAugmentation for x in ['reverse_lr']):
                    reverse_select = 1
                    augmentation_type = 'reverse_left_right'
                    reverse(result_path+'/normalization/grayscale/')
                    sample(result_path+'/augmentation/'+augmentation_type, 'gray', augmentation_type)
                elif all(x in check_dataAugmentation for x in ['reverse_tb']):    
                    reverse_select = 0
                    augmentation_type = 'reverse_top_bottom'
                    reverse(result_path+'/normalization/grayscale/')
                    sample(result_path+'/augmentation/'+augmentation_type, 'gray', augmentation_type)
                elif all(x in check_dataAugmentation for x in ['rotation_90']):
                    augmentation_type = 'rotation_90'
                    rotation(result_path+'/normalization/grayscale/')
                    sample(result_path+'/augmentation/'+augmentation_type, 'gray', augmentation_type)
                elif all(x in check_dataAugmentation for x in ['rotation_180']):
                    augmentation_type = 'rotation_180'
                    rotation(result_path+'/normalization/grayscale/')
                    sample(result_path+'/augmentation/'+augmentation_type, 'gray', augmentation_type)
                elif all(x in check_dataAugmentation for x in ['rotation_270']):
                    augmentation_type = 'rotation_270'
                    rotation(result_path+'/normalization/grayscale/')
                    sample(result_path+'/augmentation/'+augmentation_type, 'gray', augmentation_type)

            # select = 2
            elif len(check_dataAugmentation) == 2:
                if all(x in check_dataAugmentation for x in ['reverse_lr', 'reverse_tb']):
                    reverse_select = 1
                    augmentation_type = 'reverse_left_right'
                    reverse(result_path+'/normalization/grayscale/')
                    sample(result_path+'/augmentation/'+augmentation_type, 'gray', augmentation_type)
                    reverse_select = 0
                    augmentation_type = 'reverse_top_bottom'
                    reverse(result_path+'/normalization/grayscale/')
                    sample(result_path+'/augmentation/'+augmentation_type, 'gray', augmentation_type)
                elif all(x in check_dataAugmentation for x in ['reverse_lr', 'rotation_90']):
                    reverse_select = 1
                    augmentation_type = 'reverse_left_right'
                    reverse(result_path+'/normalization/grayscale/')
                    sample(result_path+'/augmentation/'+augmentation_type, 'gray', augmentation_type)
                    augmentation_type = 'rotation_90'
                    rotation(result_path+'/normalization/grayscale/')
                    sample(result_path+'/augmentation/'+augmentation_type, 'gray', augmentation_type)
                elif all(x in check_dataAugmentation for x in ['reverse_lr', 'rotation_180']):
                    reverse_select = 1
                    augmentation_type = 'reverse_left_right'
                    reverse(result_path+'/normalization/grayscale/')
                    sample(result_path+'/augmentation/'+augmentation_type, 'gray', augmentation_type)
                    augmentation_type = 'rotation_180'
                    rotation(result_path+'/normalization/grayscale/')
                    sample(result_path+'/augmentation/'+augmentation_type, 'gray', augmentation_type)
                elif all(x in check_dataAugmentation for x in ['reverse_lr', 'rotation_270']):
                    reverse_select = 1
                    augmentation_type = 'reverse_left_right'
                    reverse(result_path+'/normalization/grayscale/')
                    sample(result_path+'/augmentation/'+augmentation_type, 'gray', augmentation_type)
                    augmentation_type = 'rotation_270'
                    rotation(result_path+'/normalization/grayscale/')
                    sample(result_path+'/augmentation/'+augmentation_type, 'gray', augmentation_type)
                elif all(x in check_dataAugmentation for x in ['reverse_tb', 'rotation_90']):
                    reverse_select = 0
                    augmentation_type = 'reverse_top_bottom'
                    reverse(result_path+'/normalization/grayscale/')
                    sample(result_path+'/augmentation/'+augmentation_type, 'gray', augmentation_type)
                    augmentation_type = 'rotation_90'
                    rotation(result_path+'/normalization/grayscale/')
                    sample(result_path+'/augmentation/'+augmentation_type, 'gray', augmentation_type)
                elif all(x in check_dataAugmentation for x in ['reverse_tb', 'rotation_180']):
                    reverse_select = 0
                    augmentation_type = 'reverse_top_bottom'
                    reverse(result_path+'/normalization/grayscale/')
                    sample(result_path+'/augmentation/'+augmentation_type, 'gray', augmentation_type)
                    augmentation_type = 'rotation_180'
                    rotation(result_path+'/normalization/grayscale/')
                    sample(result_path+'/augmentation/'+augmentation_type, 'gray', augmentation_type)
                elif all(x in check_dataAugmentation for x in ['reverse_tb', 'rotation_270']):
                    reverse_select = 0
                    augmentation_type = 'reverse_top_bottom'
                    reverse(result_path+'/normalization/grayscale/')
                    sample(result_path+'/augmentation/'+augmentation_type, 'gray', augmentation_type)
                    augmentation_type = 'rotation_270'
                    rotation(result_path+'/normalization/grayscale/')
                    sample(result_path+'/augmentation/'+augmentation_type, 'gray', augmentation_type)
                elif all(x in check_dataAugmentation for x in ['rotation_90', 'rotation_180']):
                    augmentation_type = 'rotation_90'
                    rotation(result_path+'/normalization/grayscale/')
                    sample(result_path+'/augmentation/'+augmentation_type, 'gray', augmentation_type)
                    augmentation_type = 'rotation_180'
                    rotation(result_path+'/normalization/grayscale/')
                    sample(result_path+'/augmentation/'+augmentation_type, 'gray', augmentation_type)
                elif all(x in check_dataAugmentation for x in ['rotation_90', 'rotation_270']):
                    augmentation_type = 'rotation_90'
                    rotation(result_path+'/normalization/grayscale/')
                    sample(result_path+'/augmentation/'+augmentation_type, 'gray', augmentation_type)
                    augmentation_type = 'rotation_270'
                    rotation(result_path+'/normalization/grayscale/')
                    sample(result_path+'/augmentation/'+augmentation_type, 'gray', augmentation_type)
                elif all(x in check_dataAugmentation for x in ['rotation_180', 'rotation_270']):
                    augmentation_type = 'rotation_180'
                    rotation(result_path+'/normalization/grayscale/')
                    sample(result_path+'/augmentation/'+augmentation_type, 'gray', augmentation_type)
                    augmentation_type = 'rotation_270'
                    rotation(result_path+'/normalization/grayscale/')
                    sample(result_path+'/augmentation/'+augmentation_type, 'gray', augmentation_type)

            # select = 3
            elif len(check_dataAugmentation) == 3:
                if all(x in check_dataAugmentation for x in ['reverse_lr', 'reverse_tb', 'rotation_90']):
                    reverse_select = 1
                    augmentation_type = 'reverse_left_right'
                    reverse(result_path+'/normalization/grayscale/')
                    sample(result_path+'/augmentation/'+augmentation_type, 'gray', augmentation_type)
                    reverse_select = 0
                    augmentation_type = 'reverse_top_bottom'
                    reverse(result_path+'/normalization/grayscale/')
                    sample(result_path+'/augmentation/'+augmentation_type, 'gray', augmentation_type)
                    augmentation_type = 'rotation_90'
                    rotation(result_path+'/normalization/grayscale/')
                    sample(result_path+'/augmentation/'+augmentation_type, 'gray', augmentation_type)
                elif all(x in check_dataAugmentation for x in ['reverse_lr', 'reverse_tb', 'rotation_180']):
                    reverse_select = 1
                    augmentation_type = 'reverse_left_right'
                    reverse(result_path+'/normalization/grayscale/')
                    sample(result_path+'/augmentation/'+augmentation_type, 'gray', augmentation_type)
                    reverse_select = 0
                    augmentation_type = 'reverse_top_bottom'
                    reverse(result_path+'/normalization/grayscale/')
                    sample(result_path+'/augmentation/'+augmentation_type, 'gray', augmentation_type)
                    augmentation_type = 'rotation_180'
                    rotation(result_path+'/normalization/grayscale/')
                    sample(result_path+'/augmentation/'+augmentation_type, 'gray', augmentation_type)
                elif all(x in check_dataAugmentation for x in ['reverse_lr', 'reverse_tb', 'rotation_270']):
                    reverse_select = 1
                    augmentation_type = 'reverse_left_right'
                    reverse(result_path+'/normalization/grayscale/')
                    sample(result_path+'/augmentation/'+augmentation_type, 'gray', augmentation_type)
                    reverse_select = 0
                    augmentation_type = 'reverse_top_bottom'
                    reverse(result_path+'/normalization/grayscale/')
                    sample(result_path+'/augmentation/'+augmentation_type, 'gray', augmentation_type)
                    augmentation_type = 'rotation_270'
                    rotation(result_path+'/normalization/grayscale/')
                    sample(result_path+'/augmentation/'+augmentation_type, 'gray', augmentation_type)
                elif all(x in check_dataAugmentation for x in ['reverse_lr', 'rotation_90', 'rotation_180']):
                    reverse_select = 1
                    augmentation_type = 'reverse_left_right'
                    reverse(result_path+'/normalization/grayscale/')
                    sample(result_path+'/augmentation/'+augmentation_type, 'gray', augmentation_type)
                    augmentation_type = 'rotation_90'
                    rotation(result_path+'/normalization/grayscale/')
                    sample(result_path+'/augmentation/'+augmentation_type, 'gray', augmentation_type)
                    augmentation_type = 'rotation_180'
                    rotation(result_path+'/normalization/grayscale/')
                    sample(result_path+'/augmentation/'+augmentation_type, 'gray', augmentation_type)
                elif all(x in check_dataAugmentation for x in ['reverse_lr', 'rotation_90', 'rotation_270']):
                    reverse_select = 1
                    augmentation_type = 'reverse_left_right'
                    reverse(result_path+'/normalization/grayscale/')
                    sample(result_path+'/augmentation/'+augmentation_type, 'gray', augmentation_type)
                    augmentation_type = 'rotation_90'
                    rotation(result_path+'/normalization/grayscale/')
                    sample(result_path+'/augmentation/'+augmentation_type, 'gray', augmentation_type)
                    augmentation_type = 'rotation_270'
                    rotation(result_path+'/normalization/grayscale/')
                    sample(result_path+'/augmentation/'+augmentation_type, 'gray', augmentation_type)
                elif all(x in check_dataAugmentation for x in ['reverse_lr', 'rotation_180', 'rotation_270']):
                    reverse_select = 1
                    augmentation_type = 'reverse_left_right'
                    reverse(result_path+'/normalization/grayscale/')
                    sample(result_path+'/augmentation/'+augmentation_type, 'gray', augmentation_type)
                    augmentation_type = 'rotation_180'
                    rotation(result_path+'/normalization/grayscale/')
                    sample(result_path+'/augmentation/'+augmentation_type, 'gray', augmentation_type)
                    augmentation_type = 'rotation_270'
                    rotation(result_path+'/normalization/grayscale/')
                    sample(result_path+'/augmentation/'+augmentation_type, 'gray', augmentation_type)
                elif all(x in check_dataAugmentation for x in ['reverse_tb', 'rotation_90', 'rotation_180']):
                    reverse_select = 0
                    augmentation_type = 'reverse_top_bottom'
                    reverse(result_path+'/normalization/grayscale/')
                    sample(result_path+'/augmentation/'+augmentation_type, 'gray', augmentation_type)
                    augmentation_type = 'rotation_90'
                    rotation(result_path+'/normalization/grayscale/')
                    sample(result_path+'/augmentation/'+augmentation_type, 'gray', augmentation_type)
                    augmentation_type = 'rotation_180'
                    rotation(result_path+'/normalization/grayscale/')
                    sample(result_path+'/augmentation/'+augmentation_type, 'gray', augmentation_type)
                elif all(x in check_dataAugmentation for x in ['reverse_tb', 'rotation_90', 'rotation_270']):
                    reverse_select = 0
                    augmentation_type = 'reverse_top_bottom'
                    reverse(result_path+'/normalization/grayscale/')
                    sample(result_path+'/augmentation/'+augmentation_type, 'gray', augmentation_type)
                    augmentation_type = 'rotation_90'
                    rotation(result_path+'/normalization/grayscale/')
                    sample(result_path+'/augmentation/'+augmentation_type, 'gray', augmentation_type)
                    augmentation_type = 'rotation_270'
                    rotation(result_path+'/normalization/grayscale/')
                    sample(result_path+'/augmentation/'+augmentation_type, 'gray', augmentation_type)
                elif all(x in check_dataAugmentation for x in ['reverse_tb', 'rotation_180', 'rotation_270']):
                    reverse_select = 0
                    augmentation_type = 'reverse_top_bottom'
                    reverse(result_path+'/normalization/grayscale/')
                    sample(result_path+'/augmentation/'+augmentation_type, 'gray', augmentation_type)
                    augmentation_type = 'rotation_180'
                    rotation(result_path+'/normalization/grayscale/')
                    sample(result_path+'/augmentation/'+augmentation_type, 'gray', augmentation_type)
                    rotation(result_path+'/normalization/grayscale/')
                    sample(result_path+'/augmentation/'+augmentation_type, 'gray', augmentation_type)
                elif all(x in check_dataAugmentation for x in ['rotation_90', 'rotation_180', 'rotation_270']):
                    augmentation_type = 'rotation_90'
                    rotation(result_path+'/normalization/grayscale/')
                    sample(result_path+'/augmentation/'+augmentation_type, 'gray', augmentation_type)
                    augmentation_type = 'rotation_180'
                    rotation(result_path+'/normalization/grayscale/')
                    sample(result_path+'/augmentation/'+augmentation_type, 'gray', augmentation_type)
                    augmentation_type = 'rotation_270'
                    rotation(result_path+'/normalization/grayscale/')
                    sample(result_path+'/augmentation/'+augmentation_type, 'gray', augmentation_type)

            # select = 4
            elif len(check_dataAugmentation) == 4:
                if all(x in check_dataAugmentation for x in ['reverse_lr', 'reverse_tb', 'rotation_90', 'rotation_180']):
                    reverse_select = 1
                    augmentation_type = 'reverse_left_right'
                    reverse(result_path+'/normalization/grayscale/')
                    sample(result_path+'/augmentation/'+augmentation_type, 'gray', augmentation_type)
                    reverse_select = 0
                    augmentation_type = 'reverse_top_bottom'
                    reverse(result_path+'/normalization/grayscale/')
                    sample(result_path+'/augmentation/'+augmentation_type, 'gray', augmentation_type)
                    augmentation_type = 'rotation_90'
                    rotation(result_path+'/normalization/grayscale/')
                    sample(result_path+'/augmentation/'+augmentation_type, 'gray', augmentation_type)
                    augmentation_type = 'rotation_180'
                    rotation(result_path+'/normalization/grayscale/')
                    sample(result_path+'/augmentation/'+augmentation_type, 'gray', augmentation_type)
                elif all(x in check_dataAugmentation for x in ['reverse_lr', 'reverse_tb', 'rotation_90', 'rotation_270']):
                    reverse_select = 1
                    augmentation_type = 'reverse_left_right'
                    reverse(result_path+'/normalization/grayscale/')
                    sample(result_path+'/augmentation/'+augmentation_type, 'gray', augmentation_type)
                    reverse_select = 0
                    augmentation_type = 'reverse_top_bottom'
                    reverse(result_path+'/normalization/grayscale/')
                    sample(result_path+'/augmentation/'+augmentation_type, 'gray', augmentation_type)
                    augmentation_type = 'rotation_90'
                    rotation(result_path+'/normalization/grayscale/')
                    sample(result_path+'/augmentation/'+augmentation_type, 'gray', augmentation_type)
                    augmentation_type = 'rotation_270'
                    rotation(result_path+'/normalization/grayscale/')
                    sample(result_path+'/augmentation/'+augmentation_type, 'gray', augmentation_type)
                elif all(x in check_dataAugmentation for x in ['reverse_lr', 'reverse_tb', 'rotation_180', 'rotation_270']):
                    reverse_select = 1
                    augmentation_type = 'reverse_left_right'
                    reverse(result_path+'/normalization/grayscale/')
                    sample(result_path+'/augmentation/'+augmentation_type, 'gray', augmentation_type)
                    reverse_select = 0
                    augmentation_type = 'reverse_top_bottom'
                    reverse(result_path+'/normalization/grayscale/')
                    sample(result_path+'/augmentation/'+augmentation_type, 'gray', augmentation_type)
                    augmentation_type = 'rotation_180'
                    rotation(result_path+'/normalization/grayscale/')
                    sample(result_path+'/augmentation/'+augmentation_type, 'gray', augmentation_type)
                    augmentation_type = 'rotation_270'
                    rotation(result_path+'/normalization/grayscale/')
                    sample(result_path+'/augmentation/'+augmentation_type, 'gray', augmentation_type)
                elif all(x in check_dataAugmentation for x in ['reverse_lr', 'rotation_90', 'rotation_180', 'rotation_270']):
                    reverse_select = 1
                    augmentation_type = 'reverse_left_right'
                    reverse(result_path+'/normalization/grayscale/')
                    sample(result_path+'/augmentation/'+augmentation_type, 'gray', augmentation_type)
                    augmentation_type = 'rotation_90'
                    rotation(result_path+'/normalization/grayscale/')
                    sample(result_path+'/augmentation/'+augmentation_type, 'gray', augmentation_type)
                    augmentation_type = 'rotation_180'
                    rotation(result_path+'/normalization/grayscale/')
                    sample(result_path+'/augmentation/'+augmentation_type, 'gray', augmentation_type)
                    augmentation_type = 'rotation_270'
                    rotation(result_path+'/normalization/grayscale/')
                    sample(result_path+'/augmentation/'+augmentation_type, 'gray', augmentation_type)
                elif all(x in check_dataAugmentation for x in ['reverse_tb', 'rotation_90', 'rotation_180', 'rotation_270']):
                    reverse_select = 0
                    augmentation_type = 'reverse_top_bottom'
                    reverse(result_path+'/normalization/grayscale/')
                    sample(result_path+'/augmentation/'+augmentation_type, 'gray', augmentation_type)
                    augmentation_type = 'rotation_90'
                    rotation(result_path+'/normalization/grayscale/')
                    sample(result_path+'/augmentation/'+augmentation_type, 'gray', augmentation_type)
                    augmentation_type = 'rotation_180'
                    rotation(result_path+'/normalization/grayscale/')
                    sample(result_path+'/augmentation/'+augmentation_type, 'gray', augmentation_type)
                    augmentation_type = 'rotation_270'
                    rotation(result_path+'/normalization/grayscale/')
                    sample(result_path+'/augmentation/'+augmentation_type, 'gray', augmentation_type)

            # select = 5
            elif len(check_dataAugmentation) == 5:
                if all(x in check_dataAugmentation for x in ['reverse_lr', 'reverse_tb', 'rotation_90', 'rotation_180', 'rotation_270']):
                    reverse_select = 1
                    augmentation_type = 'reverse_left_right'
                    reverse(result_path+'/normalization/grayscale/')
                    sample(result_path+'/augmentation/'+augmentation_type, 'gray', augmentation_type)
                    reverse_select = 0
                    augmentation_type = 'reverse_top_bottom'
                    reverse(result_path+'/normalization/grayscale/')
                    sample(result_path+'/augmentation/'+augmentation_type, 'gray', augmentation_type)
                    augmentation_type = 'rotation_90'
                    rotation(result_path+'/normalization/grayscale/')
                    sample(result_path+'/augmentation/'+augmentation_type, 'gray', augmentation_type)
                    augmentation_type = 'rotation_180'
                    rotation(result_path+'/normalization/grayscale/')
                    sample(result_path+'/augmentation/'+augmentation_type, 'gray', augmentation_type)
                    augmentation_type = 'rotation_270'
                    rotation(result_path+'/normalization/grayscale/')
                    sample(result_path+'/augmentation/'+augmentation_type, 'gray', augmentation_type)

            elif len(check_dataAugmentation) == 0:
                normalization_type = 'grayscale'
                grayscale()
                sample(result_path+'/normalization/'+normalization_type, 'gray', normalization_type)

#        else:
#            sample(tmp_image_path, 'color', 'raw')

    ################################################################################################
    ## preprocessing task 1, 2, 3, 4 run
    ################################################################################################

    #db = pymysql.connect(
    #    host = '10.40.217.236',
    #    user = 'root',
    #    password = 'password',
    #    port = 3306,
    #    db = 'yolo',
    #    charset = 'utf8'
    #)
    db = pymysql.connect(host='10.40.217.236', user='root', password='password', port=3307, db='sms', charset='utf8')

    def db_mysql_stat_update(projectId, versionId, statusOfPreprocessing):
        cursor = db.cursor()
        sql = 'Update Stat set statusOfPreprocessing=%s where (projectId, versionId)=%s'
        val = [statusOfPreprocessing, (projectId, versionId)]
        cursor.execute(sql, val)
        db.commit()
        cursor.close()   

    def db_mysql_preprocessing_update(projectId, versionId, numOfTrain, numOfTest, numOfValidation, numOfRaw, numOfAugmentation, numOfAugmentationRaw):
        cursor = db.cursor()
        sql = 'Update Preprocessing set numOfTrain=%s, numOfTest=%s, numOfValidation=%s, numOfRaw=%s, numOfAugmentation=%s, numOfAugmentationRaw=%s where (projectId, versionId)=%s'
        val = [numOfTrain, numOfTest, numOfValidation, numOfRaw, numOfAugmentation, numOfAugmentationRaw, (projectId, versionId)]
        cursor.execute(sql, val)
        db.commit()
        cursor.close()

    while True:

        try:
            db_mysql_stat_update(
                projectId = projectId, 
                versionId = versionId, 
                statusOfPreprocessing = 'RUNNING'
            )
            preprocessings()
            os.makedirs(tmps, exist_ok=True)
            os.makedirs(tmps_split, exist_ok=True)
            for (root, directories, files) in os.walk(result_path+'/normalization'):
                for file in files:
                    file_path = os.path.join(root, file)
                    shutil.copy2(file_path, tmps)
            for (root, directories, files) in os.walk(result_path+'/augmentation'):
                for file in files:
                    file_path = os.path.join(root, file)
                    shutil.copy2(file_path, tmps)


            trainRatios = trainRatio/100
            validationRatios = validationRatio/100
            testRatios = testRatio/100
            splitfolders.ratio(result_path+'/tmp', output=tmps_split, seed=2024, ratio=(trainRatios, validationRatios, testRatios), group_prefix=2)
            for i in os.listdir(tmps_split+'/train/a'):
                client.fput_object(bucket_name=bucket, object_name=projectId+'/'+versionId+'/preprocessing/datasplit/train/'+i, file_path=tmps_split+'/train/a/'+i)
            for i in os.listdir(tmps_split+'/val/a'):
                client.fput_object(bucket_name=bucket, object_name=projectId+'/'+versionId+'/preprocessing/datasplit/val/'+i, file_path=tmps_split+'/val/a/'+i)
            for i in os.listdir(tmps_split+'/test/a'):
                client.fput_object(bucket_name=bucket, object_name=projectId+'/'+versionId+'/preprocessing/datasplit/test/'+i, file_path=tmps_split+'/test/a/'+i)
            raw_count = sorted([f for f in os.listdir(tmp_image_path) if f.endswith(tuple(['jpg', 'png', 'JPG', 'PNG']))])
            train_count = sorted([f for f in os.listdir(tmps_split+'/train/a') if f.endswith(tuple(['jpg', 'png', 'JPG', 'PNG']))])
            print('train count', len(train_count))
            val_count = sorted([f for f in os.listdir(tmps_split+'/val/a') if f.endswith(tuple(['jpg', 'png', 'JPG', 'PNG']))])
            print('validation count', len(val_count))
            test_count = sorted([f for f in os.listdir(tmps_split+'/test/a') if f.endswith(tuple(['jpg', 'png', 'JPG', 'PNG']))])
            print('test count', len(test_count))
            break

#        except:
#            db_mysql_stat_update(
#                projectId = projectId,
#                versionId = versionId,
#                statusOfPreprocessing = 'ERROR'
#            )
#            break

        finally:
            number_of_raw = len(raw_count)            
            number_of_train = len(train_count)
            number_of_validation = len(val_count)
            number_of_test = len(test_count)
            if dataNormalization == 'grayscale':
                number_of_aug = (number_of_train+number_of_validation+number_of_test)-number_of_raw
                number_of_aug_raw = number_of_raw+number_of_aug
            db_mysql_preprocessing_update(
                projectId = projectId,
                versionId = versionId,
                numOfTrain = number_of_train, 
                numOfTest = number_of_test, 
                numOfValidation = number_of_validation, 
                numOfRaw = number_of_raw, 
                numOfAugmentation = number_of_aug, 
                numOfAugmentationRaw = number_of_aug_raw
            )
            db_mysql_stat_update(
                projectId = projectId,
                versionId = versionId,
                statusOfPreprocessing = 'FINISH'
            )            
            data_train = result_path+'/datasplit/train'
            data_val = result_path+'/datasplit/val'
            data_test = result_path+'/datasplit/test'
            os.makedirs(data_train, exist_ok=True)
            os.makedirs(data_val, exist_ok=True)
            os.makedirs(data_test, exist_ok=True)
            for (root, directories, files) in os.walk(tmps_split+'/train'):
                for file in files:
                    file_path = os.path.join(root, file)
                    shutil.copy2(file_path, data_train)
            for (root, directories, files) in os.walk(tmps_split+'/val'):
                for file in files:
                    file_path = os.path.join(root, file)
                    shutil.copy2(file_path, data_val)
            for (root, directories, files) in os.walk(tmps_split+'/test'):
                for file in files:
                    file_path = os.path.join(root, file)
                    shutil.copy2(file_path, data_test)
            shutil.rmtree(tmps)
            shutil.rmtree(tmps_split)
            shutil.rmtree(result_path+'/tmp')

################################################################################################
## kubeflow pipeline upload
################################################################################################

def list_type(v):
    return re.sub(r'[\[\]]', '', v)

def pipelines():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--projectId', type=str)
    parser.add_argument('--versionId', type=str)
    parser.add_argument('--dataPath', type=list_type)
    parser.add_argument('--dataNormalization', type=list_type)
    parser.add_argument('--dataAugmentation', type=list_type)
    parser.add_argument('--trainRatio', type=int)
    parser.add_argument('--validationRatio', type=int)
    parser.add_argument('--testRatio', type=int)
    args = parser.parse_args()

    print(args.dataAugmentation)
    print(type(args.dataAugmentation))

    Preprocessing_apply = Preprocessing(args.projectId, args.versionId, args.dataPath, args.dataNormalization, args.dataAugmentation, args.trainRatio, args.validationRatio, args.testRatio) \
        .set_display_name('Data Preprocessing') \
        .apply(onprem.mount_pvc('dlabflow-claim-test', volume_name='data', volume_mount_path='/mnt/dlabflow'))
    Preprocessing_apply.execution_options.caching_strategy.max_cache_staleness = 'P0D'

if __name__ == '__main__':
    pipeline_package_path = 'preprocessing_pipelines.zip'
    kfp.compiler.Compiler().compile(pipelines, pipeline_package_path)
    HOST = 'http://10.40.217.236:8080/'
    #USERNAME = 'user@example.com'
    #PASSWORD = '12341234'
    #NAMESPACE = 'kubeflow-user-example-com'
    USERNAME = 'kubeflow-grit-test@service.com'
    PASSWORD = 'N2aUQEQbhF09WFc'
    NAMESPACE = 'kubeflow-grit-test'    
    session = requests.Session()
    response = session.get(HOST)
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    data = {'login': USERNAME, 'password': PASSWORD}
    session.post(response.url, headers = headers, data = data)
    session_cookie = session.cookies.get_dict()['authservice_session']
    client = kfp.Client(host = f'{HOST}/pipeline', cookies = f'authservice_session={session_cookie}', namespace = NAMESPACE)
    experiment = client.create_experiment(name='Preprocessing')
    run = client.run_pipeline(experiment.id, 'Preprocessing pipelines', pipeline_package_path)
