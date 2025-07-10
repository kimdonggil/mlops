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
from pydantic import BaseModel
import typing as t
import requests

################################################################################################
## kubeflow pipeline
################################################################################################

@partial(create_component_from_func, base_image='dgkim1983/dlabflow:yolo-24061401', packages_to_install=['minio', 'bentoml', 'pymysql'])
def Inference(projectId: str, versionId: str, inferenceAlgorithm: str):
    from ultralytics import YOLO
    from sklearn.model_selection import train_test_split
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    import shutil
    from pathlib import Path
    import xml.etree.ElementTree as ET
    import decimal
    import os
    import datetime
    import glob
    import tqdm
    import random
    import math
    from minio import Minio
    import bentoml
    import csv
    from pathlib import Path
    import pymysql
    import sys
    import time
    import datetime
    import torch

    ################################################################################################
    ## data path
    ################################################################################################

    bucket = 'aiproject'    
    minio_path = '/mnt/dlabflow/backend/minio/'+bucket
    train_path = minio_path+'/'+projectId+'/'+versionId+'/train'
    inference_path = minio_path+'/'+projectId+'/'+versionId+'/inference'
    inference_before_path = inference_path+'/before'
    os.makedirs(inference_before_path, exist_ok=True)
    inference_after_path = inference_path+'/after'
    os.makedirs(inference_after_path, exist_ok=True)

    client = Minio(endpoint='10.40.217.236:9002', access_key='dlab-backend', secret_key='dlab-backend-secret', secure=False)
    for item in client.list_objects(bucket_name=bucket, prefix=projectId+'/'+versionId+'/inference/before', recursive=True):
        client.fget_object(bucket_name=bucket, object_name=item.object_name, file_path=minio_path+'/'+item.object_name)

    ################################################################################################
    ## model task 1 : predict
    ################################################################################################

    def predict():
        cuda = torch.cuda.is_available()
        print(cuda)
        if cuda == True:
            for item in client.list_objects(bucket_name=bucket, prefix=projectId+'/'+versionId+'/train/model/train/weight', recursive=True):
                client.fget_object(bucket_name=bucket, object_name=item.object_name, file_path=inference_path+'/algorithm/'+item.object_name.split('/')[-1])

            if inferenceAlgorithm == 'yolo_version_5_normal':
                model = YOLO(inference_path+'/algorithm/best.pt')
            elif inferenceAlgorithm == 'yolo_version_5_small':
                model = YOLO(inference_path+'/algorithm/best.pt')
            elif inferenceAlgorithm == 'yolo_version_5_medium':
                model = YOLO(inference_path+'/algorithm/best.pt')
            elif inferenceAlgorithm == 'yolo_version_5_large':
                model = YOLO(inference_path+'/algorithm/best.pt')
            elif inferenceAlgorithm == 'yolo_version_5_xlarge':
                model = YOLO(inference_path+'/algorithm/best.pt')
            elif inferenceAlgorithm == 'yolo_version_8_normal':
                model = YOLO(inference_path+'/algorithm/best.pt')
            elif inferenceAlgorithm == 'yolo_version_8_small':
                model = YOLO(inference_path+'/algorithm/best.pt')
            elif inferenceAlgorithm == 'yolo_version_8_medium':
                model = YOLO(inference_path+'/algorithm/best.pt')
            elif inferenceAlgorithm == 'yolo_version_8_large':
                model = YOLO(inference_path+'/algorithm/best.pt')
            elif inferenceAlgorithm == 'yolo_version_8_xlarge':
                model = YOLO(inference_path+'/algorithm/best.pt')

            print(model)

            result = model.predict(source=inference_before_path, save=True, project=inference_path, name='after', exist_ok=True)
            print(result)

            for i in os.listdir(inference_after_path):
                client.fput_object(bucket_name=bucket, object_name=projectId+'/'+versionId+'/inference/after/'+i, file_path=inference_after_path+'/'+i)            
        else:
            print('GPU is not using')

    ################################################################################################
    ## preprocessing task 1 run
    ################################################################################################

    def db_mysql_stat_update(sql_select, projectId, versionId, statusOfInference):
        db = pymysql.connect(host='10.40.217.236', user='root', password='password', port=3306, db='yolo', charset='utf8')
        cursor = db.cursor()
        sql = f"Update {sql_select} set statusOfInference=%s where (projectId, versionId)=%s"
        val = [statusOfInference, (projectId, versionId)]
        cursor.execute(sql, val)
        db.commit()
        cursor.close()    

    def db_mysql_inference_update(sql_select, projectId, versionId, inferenceResult):
        db = pymysql.connect(host='10.40.217.236', user='root', password='password', port=3306, db='yolo', charset='utf8')
        cursor = db.cursor()
        sql = f"Update {sql_select} set inferenceResult=%s where (projectId, versionId)=%s"
        val = [inferenceResult, (projectId, versionId)]
        cursor.execute(sql, val)
        db.commit()
        cursor.close()

    while True:
        try:
            status_of_inference = 'Running'
            db_mysql_stat_update(sql_select='Stat', projectId=projectId, versionId=versionId, statusOfInference=status_of_inference)
            predict()
            break
        except:
            status_of_inference = 'Error'
            db_mysql_stat_update(sql_select='Stat', projectId=projectId, versionId=versionId, statusOfInference=status_of_inference)
            break
        finally:
            status_of_inference = 'Finish'
            db_mysql_stat_update(sql_select='Stat', projectId=projectId, versionId=versionId, statusOfInference=status_of_inference)
            db_mysql_inference_update(sql_select='Inference', projectId=projectId, versionId=versionId, inferenceResult=f"minio://aiproject/{projectId}/{versionId}/inference/after/")

            object_name = f'{projectId}_{versionId}.csv'
            client.remove_object(bucket, object_name)

#            shutil.rmtree(minio_path)

################################################################################################
## kubeflow pipeline upload
################################################################################################

def pipelines():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--projectId', type=str)
    parser.add_argument('--versionId', type=str)
    parser.add_argument('--inferenceAlgorithm', type=str)
    args = parser.parse_args()

    Inference_apply = Inference(args.projectId, args.versionId, args.inferenceAlgorithm) \
        .set_display_name('Model Inference') \
        .apply(onprem.mount_pvc('dlabflow-claim', volume_name='data', volume_mount_path='/mnt/dlabflow'))
    smh_vol = kfp.dsl.PipelineVolume(name = 'shm-vol', empty_dir = {'medium': 'Memory'})
    Inference_apply.add_pvolumes({'/dev/shm': smh_vol})        
    Inference_apply.execution_options.caching_strategy.max_cache_staleness = 'P0D'

if __name__ == '__main__':
    pipeline_package_path = 'inference_pipelines.zip'
    kfp.compiler.Compiler().compile(pipelines, pipeline_package_path)
    HOST = 'http://10.40.217.236:8080/'
    #USERNAME = 'user@example.com'
    #PASSWORD = '12341234'
    #NAMESPACE = 'kubeflow-user-example-com'
    USERNAME = 'kubeflow-grit@service.com'
    PASSWORD = 'kubeflow-grit-secret'
    NAMESPACE = 'grit'    
    session = requests.Session()
    response = session.get(HOST)
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    data = {'login': USERNAME, 'password': PASSWORD}
    session.post(response.url, headers = headers, data = data)
    session_cookie = session.cookies.get_dict()['authservice_session']
    client = kfp.Client(host = f'{HOST}/pipeline', cookies = f'authservice_session={session_cookie}', namespace = NAMESPACE)
    experiment = client.create_experiment(name='Inference')
    run = client.run_pipeline(experiment.id, 'Inference pipelines', pipeline_package_path)

    
