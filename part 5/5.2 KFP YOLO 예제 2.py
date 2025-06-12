from functools import partial
import kfp
from kfp import onprem
from kfp.components import create_component_from_func
from kubernetes.client import V1EnvVar
from kfp.dsl import PipelineVolume

@partial(create_component_from_func, base_image='python:3.10', packages_to_install=['minio'])
def minio_connect():
    from minio import Minio
    import os
    bucket = 'data'
    client = Minio(endpoint='10.40.217.244:9000', access_key='minio', secret_key='minio123', secure=False)
    folders = ['example/train', 'example/valid', 'example/test']
    local_base = f"/mnt/working/kubeflow/volumes/{bucket}"
    try:
        for folder in folders:
            objects = client.list_objects(bucket, prefix=folder, recursive=True)
            for obj in objects:
                local_path = os.path.join(local_base, obj.object_name)
                dir_path = os.path.dirname(local_path)
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path, exist_ok=True)
                client.fget_object(bucket, obj.object_name, local_path)
        print(f"All files downloaded successfully to {local_base}")
    except Exception as e:
        print(f"Data download failed: {e}")
        raise

@partial(create_component_from_func, base_image='ultralytics/ultralytics:latest')
def model_training(algorithm: str, batch: int, epoch: int):
    import yaml
    import torch
    from ultralytics import YOLO
    bucket = 'data'
    data_config = {
        'train': f'/mnt/working/kubeflow/volumes/{bucket}/example/train',
        'val': f'/mnt/working/kubeflow/volumes/{bucket}/example/valid',
        'test': f'/mnt/working/kubeflow/volumes/{bucket}/example/test',
        'nc': 19,
        'names': [
            'car', 'tvmonitor', 'aeroplane', 'cat', 'motorbike', 'bird', 'bicycle',
            'chair', 'cow', 'train', 'horse', 'sofa', 'diningtable', 'boat',
            'pottedplant', 'bottle', 'person', 'bus', 'dog'
        ]
    }
    with open('custom.yaml', 'w') as f:
        yaml.dump(data_config, f)
    device = '0' if torch.cuda.is_available() else 'cpu'
    try:
        model = YOLO(algorithm)
        model.train(data='custom.yaml', batch=batch, epochs=epoch, device=device)
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise

@kfp.dsl.pipeline(name='Example Pipeline', description='Train YOLO model using data from MinIO')
def example_pipeline(algorithm: str, batch: int, epoch: int):
    minio_task = minio_connect() \
        .apply(onprem.mount_pvc('example-claim', volume_name='data', volume_mount_path='/mnt/working/kubeflow/volumes'))
    train_task = model_training(algorithm, batch, epoch) \
        .apply(onprem.mount_pvc('example-claim', volume_name='data', volume_mount_path='/mnt/working/kubeflow/volumes')) \
        .add_env_variable(V1EnvVar(name='CUDA_VISIBLE_DEVICES', value='0'))
    shm_vol = PipelineVolume(name='shm-vol', empty_dir={'medium': 'Memory'})
    train_task.add_pvolumes({'/dev/shm': shm_vol})
    train_task.after(minio_task)

if __name__ == '__main__':
    kfp.compiler.Compiler().compile(example_pipeline, 'example_pipeline.yaml')