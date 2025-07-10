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
from kubernetes.client import V1EnvVar

################################################################################################
## kubeflow pipeline
################################################################################################

@partial(create_component_from_func, base_image='dgkim1983/dlabflow:model-20250304', packages_to_install=['minio', 'bentoml', 'pymysql'])
def Inference(projectId: str, versionId: str, sessionId: str):
        import random
        import os
        import glob
        import pandas as pd
        import xml.etree.ElementTree as ET
        import git
        import wget
        import subprocess
        import re
        import tarfile
        import shutil
        import pathlib
        import matplotlib
        import matplotlib.pyplot as plt
        import io
        import scipy.misc
        import ipywidgets as widgets
        from IPython.display import display
        import numpy as np; print('numpy version: ', np.__version__)
        from six import BytesIO
        from PIL import Image, ImageDraw, ImageFont
        from six.moves.urllib.request import urlopen
        import tensorflow as tf; print('tensorflow version: ', tf.__version__)
        import tensorflow as tf_hub; print('tensorflow hub version: ', tf_hub.__version__)
        import keras; print('keras version: ', keras.__version__)
        from tensorboard import notebook

    ################################################################################################
    ## data path
    ################################################################################################

    bucket = 'aiproject'    
    minio_path = '/mnt/dlabflow/backend/minio/'+bucket
    train_path = minio_path+'/'+projectId+'/'+versionId+'/train'
    inference_path = minio_path+'/'+projectId+'/'+versionId+'/inference'
    inference_before_path = inference_path+'/'+sessionId+'/before'
    os.makedirs(inference_before_path, exist_ok=True)
    inference_after_path = inference_path+'/'+sessionId
    os.makedirs(inference_after_path, exist_ok=True)
    result_path = minio_path+'/'+projectId+'/'+versionId+'/train'

    client = Minio(endpoint='10.40.217.236:9002', access_key='dlab-backend', secret_key='dlab-backend-secret', secure=False)
    for item in client.list_objects(bucket_name=bucket, prefix=projectId+'/'+versionId+'/inference/'+sessionId+'/before', recursive=True):
        client.fget_object(bucket_name=bucket, object_name=item.object_name, file_path=minio_path+'/'+item.object_name)

    ################################################################################################
    ## model task 1 : predict
    ################################################################################################

    def predict():
        print('')
        print('+-'*20 + '+')
        print('신규 데이터 추론 결과 저장')
        print('+-'*20 + '+')
        print('')

        for item in client.list_objects(bucket_name=bucket, prefix=projectId+'/train/model/train/weight', recursive=True):
            file_name = item.object_name.split('/')[-1]

            if file_name.startswith("variables"):
                save_path = f"{inference_path}/{sessionId}/save/saved_model/variables/{file_name}"
            else:
                save_path = f"{inference_path}/{sessionId}/save/saved_model/{file_name}"

            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            client.fget_object(bucket_name=bucket, object_name=item.object_name, file_path=save_path)
            print(f"Downloaded: {save_path}")

        predict_dir = os.path.join(training_path, 'after')

        os.makedirs(predict_dir, exist_ok=True)

        PATH_TO_SAVED_MODEL = inference_path+'/'+sessionId+'/save/saved_model'

        PATH_TO_INFERENCE_IMAGE = inference_before_path

        detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)

        label_map_path = result_path + '/images/labelmap.pbtxt'
        category_index = label_map_util.create_category_index_from_labelmap(label_map_path, use_display_name=True)

        image_files = [f for f in os.listdir(PATH_TO_INFERENCE_IMAGE) if f.endswith('.jpg') or f.endswith('.png')]

        for image_file in image_files:
            image_path = os.path.join(PATH_TO_INFERENCE_IMAGE, image_file)
            image_np = np.array(np.asarray(Image.open(image_path)))

            input_tensor = tf.convert_to_tensor(image_np)
            input_tensor = input_tensor[tf.newaxis,...]

            detections = detect_fn(input_tensor)

            viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np,
                detections['detection_boxes'][0].numpy(),
                detections['detection_classes'][0].numpy().astype(np.int32),
                detections['detection_scores'][0].numpy(),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=4
            )

            result_image_path = os.path.join(predict_dir, image_file)
            result_image = Image.fromarray(image_np)
            result_image.save(result_image_path)

        for i in os.listdir(inference_after_path):
            client.fput_object(bucket_name=bucket, object_name=projectId+'/'+versionId+'/inference/'+sessionId+'/after/'+i, file_path=inference_after_path+'/'+i)            
        

    ################################################################################################
    ## preprocessing task 1 run
    ################################################################################################

    db = pymysql.connect(host='10.40.217.236', user='root', password='password', port=3306, db='yolo', charset='utf8')

    def db_mysql_stat_update(sql_select, projectId, versionId, statusOfInference):
        cursor = db.cursor()
        try:
            sql = f"Update {sql_select} set statusOfInference=%s where (projectId, versionId)=%s"
            val = [statusOfInference, (projectId, versionId)]
            cursor.execute(sql, val)
            db.commit()
        finally:
            cursor.close()
            db.close()

    def db_mysql_inference_update(sql_select, projectId, versionId, statusOfInference):
        cursor = db.cursor()
        try:
            sql = f"Update {sql_select} set statusOfInference=%s where (projectId, versionId)=%s"
            val = [statusOfInference, (projectId, versionId)]
            cursor.execute(sql, val)
            db.commit()
        finally:
            cursor.close()
            db.close()

    while True:
        try:
            status_of_inference = 'RUNNING'
#            db_mysql_stat_update(sql_select='Stat', projectId=projectId, versionId=versionId, statusOfInference=status_of_inference)
            db_mysql_inference_update(sql_select='Inference', projectId=projectId, versionId=versionId, statusOfInference=status_of_inference)
            predict()
            break
        except:
            status_of_inference = 'ERROR'
#            db_mysql_stat_update(sql_select='Stat', projectId=projectId, versionId=versionId, statusOfInference=status_of_inference)
            db_mysql_inference_update(sql_select='Inference', projectId=projectId, versionId=versionId, statusOfInference=status_of_inference)
            break
        finally:
            status_of_inference = 'FINISH'
#            db_mysql_stat_update(sql_select='Stat', projectId=projectId, versionId=versionId, statusOfInference=status_of_inference)
#            db_mysql_inference_update(sql_select='Inference', projectId=projectId, versionId=versionId, inferenceResult=f"minio://aiproject/{projectId}/{versionId}/inference/{sessionId}/after/")
            db_mysql_inference_update(sql_select='Inference', projectId=projectId, versionId=versionId, statusOfInference=status_of_inference)

            object_name = f'{projectId}_{versionId}.csv'
            client.remove_object(bucket, object_name)

            before_keep = f'{projectId}/{versionId}/inference/{sessionId}/before/.keep'
            client.remove_object(bucket, before_keep)

            after_keep = f'{projectId}/{versionId}/inference/{sessionId}/after/.keep'
            client.remove_object(bucket, after_keep)

#            shutil.rmtree(minio_path)


################################################################################################
## kubeflow pipeline upload
################################################################################################

def pipelines():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--projectId', type=str)
    parser.add_argument('--versionId', type=str)
    parser.add_argument('--sessionId', type=str)
    args = parser.parse_args()

    Inference_apply = Inference(args.projectId, args.versionId, args.sessionId) \
        .set_display_name('Model Inference') \
        .apply(onprem.mount_pvc('dlabflow-claim', volume_name='data', volume_mount_path='/mnt/dlabflow'))\
        .add_env_variable(V1EnvVar(name="CUDA_VISIBLE_DEVICES", value="0"))

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
    USERNAME = 'kubeflow-grit-admin@service.com'
    PASSWORD = 'AW8QHDbX1UgyKSC'
    NAMESPACE = 'kubeflow-grit-admin'    
    session = requests.Session()
    response = session.get(HOST)
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    data = {'login': USERNAME, 'password': PASSWORD}
    session.post(response.url, headers = headers, data = data)
    session_cookie = session.cookies.get_dict()['authservice_session']
    client = kfp.Client(host = f'{HOST}/pipeline', cookies = f'authservice_session={session_cookie}', namespace = NAMESPACE)
    experiment = client.create_experiment(name='Inference')
    run = client.run_pipeline(experiment.id, 'Inference pipelines', pipeline_package_path)

    
