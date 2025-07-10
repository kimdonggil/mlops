from functools import partial
from kfp.components import create_component_from_func
from kfp import compiler, dsl, onprem
from kubernetes.client import V1EnvVar
import argparse
import requests
import kfp

#@partial(create_component_from_func, base_image='dgkim1983/dlabflow:model-20250408', packages_to_install=['minio', 'bentoml', 'pymysql'])
@partial(create_component_from_func, base_image='dgkim1983/dlabflow:model-20250408')
def Inference(projectId: str, versionId: str, sessionId: str):
    import os
    from datetime import datetime
    from pathlib import Path
    from minio import Minio
    from ultralytics import YOLO
    import pymysql
    import torch

    # Setup
    bucket = 'aiproject'
    minio_path = f'/mnt/dlabflow/backend/minio/{bucket}'
    inference_before_path = f'{minio_path}/{projectId}/{versionId}/inference/{sessionId}/before'
    inference_after_path = f'{minio_path}/{projectId}/{versionId}/inference/{sessionId}/after'
    os.makedirs(inference_before_path, exist_ok=True)
    os.makedirs(inference_after_path, exist_ok=True)

    # MinIO client
    client = Minio(
        endpoint='10.40.217.236:9002',
        access_key='dlab-backend',
        secret_key='dlab-backend-secret',
        secure=False
    )

    # Download inference input
    for item in client.list_objects(bucket, prefix=f'{projectId}/{versionId}/inference/{sessionId}/before', recursive=True):
        client.fget_object(bucket, item.object_name, f'{minio_path}/{item.object_name}')

    # Run YOLO prediction
    def predict():
        if torch.cuda.is_available():
            model_dir = f'{minio_path}/{projectId}/{versionId}/inference/{sessionId}/algorithm'
            os.makedirs(model_dir, exist_ok=True)

            for item in client.list_objects(bucket, prefix=f'{projectId}/{versionId}/train/model/train/weight', recursive=True):
                client.fget_object(bucket, item.object_name, f'{model_dir}/{Path(item.object_name).name}')

            model = YOLO(f'{model_dir}/best.pt')
            model.predict(source=inference_before_path, save=True, project=f'{minio_path}/{projectId}/{versionId}/inference/{sessionId}', name='after', exist_ok=True)

            for file in os.listdir(inference_after_path):
                client.fput_object(bucket, f'{projectId}/{versionId}/inference/{sessionId}/after/{file}', f'{inference_after_path}/{file}')
        else:
            print('[ERROR] GPU is not available.')

    # MySQL update helper
    def update_inference_status(status: str):
        conn = pymysql.connect(host='10.40.217.236', user='root', password='password', db='yolo', port=3306)
        try:
            with conn.cursor() as cursor:
                sql = "UPDATE Inference SET statusOfInference=%s WHERE projectId=%s AND versionId=%s"
                cursor.execute(sql, (status, projectId, versionId))
            conn.commit()
        finally:
            conn.close()

    # Execution flow
    try:
        #start_time = datetime.now()
        #print(f'[INFO] Inference started at {start_time.strftime("%Y-%m-%d %H:%M:%S")}')        
        update_inference_status('RUNNING')
        predict()
        #end_time = datetime.now()
        #duration = end_time - start_time
        #print(f'[INFO] Inference finished at {end_time.strftime("%Y-%m-%d %H:%M:%S")}')
        #print(f'[INFO] Total inference duration: {duration}')
        update_inference_status('FINISH')
    except Exception as e:
        print(f'[ERROR] Inference failed: {e}')
        update_inference_status('ERROR')
    finally:
        # Clean up unnecessary files in bucket
        cleanup_keys = [
            f'{projectId}_{versionId}.csv',
            f'{projectId}/{versionId}/inference/{sessionId}/before/.keep',
            f'{projectId}/{versionId}/inference/{sessionId}/after/.keep'
        ]
        for key in cleanup_keys:
            try:
                client.remove_object(bucket, key)
            except Exception as e:
                print(f'[WARN] Failed to remove {key}: {e}')

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
