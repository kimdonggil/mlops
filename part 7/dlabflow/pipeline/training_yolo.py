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

#@partial(create_component_from_func, base_image='dgkim1983/dlabflow:yolo-24061401', packages_to_install=['minio', 'bentoml', 'pymysql'])
@partial(create_component_from_func, base_image='dgkim1983/dlabflow:model-20250304', packages_to_install=['minio', 'bentoml', 'pymysql'])
def Training(projectId: str, versionId: str, algorithm:str, batchsize:int, epoch:int):
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
    from datetime import datetime
    import logging
    import torch
    import subprocess

    ################################################################################################
    ## data path
    ################################################################################################

    bucket = 'aiproject'    
    minio_path = '/mnt/dlabflow/backend/minio/'+bucket
    preprocessing_path = minio_path+'/'+projectId+'/'+versionId+'/preprocessing'
    client = Minio(endpoint='10.40.217.236:9002', access_key='dlab-backend', secret_key='dlab-backend-secret', secure=False)
    result_path = minio_path+'/'+projectId+'/'+versionId+'/train'
    training_executed_path = result_path+f"/app/{projectId}_{versionId}.txt"

    ################################################################################################
    ## model task 1 : yolo
    ################################################################################################

    def sample(data):
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
            ax.imshow(pic)
            ax.axis('off')
        else:
            [ax.set_visible(False) for ax in axes.flatten()[idx+1:]]
        fig.savefig(result_path+'/validation_result.jpg', dpi=300)

    def yolo():
        annotation_list = []
        for (root, directories, files) in os.walk(preprocessing_path):
            for file in files:
                if file.endswith('xml'):
                    annotation = os.path.join(root, file)
                    annotation_list.append(annotation)
        class_list = []
        for i in annotation_list:
            tree = ET.parse(i)
            root = tree.getroot()
            for text in root.findall('object'):
                name = text.find('name').text
                class_list.append(name)
                class_lists = list(set(class_list))
        class_id = {string : i for i, string in enumerate(class_lists)}
        data_dict = {'filename':[], 'label':[], 'class_id':[], 'width':[], 'height':[], 'bboxes':[]}
        for i in annotation_list:
            tree = ET.parse(i)
            root = tree.getroot()
            filename = root.find('filename').text
            for obj in root.findall('object'):
                label = obj.find('name').text
                bbox = []
                bndbox_tree = obj.find('bndbox')
                bbox.append(int(bndbox_tree.find('xmin').text))
                bbox.append(int(bndbox_tree.find('ymin').text))
                bbox.append(int(bndbox_tree.find('xmax').text))
                bbox.append(int(bndbox_tree.find('ymax').text))
                size = root.find('size')
                data_dict['filename'].append(filename)
                data_dict['width'].append(int(size.find('width').text))
                data_dict['height'].append(int(size.find('height').text))
                data_dict['label'].append(label)
                data_dict['class_id'].append(class_id[label])
                data_dict['bboxes'].append(bbox)
        df_data = pd.DataFrame(data_dict)
        print(df_data)
        classes = list(df_data.label.unique())
        class_count = len(classes)
        train_path = result_path+'/data/train'
        val_path = result_path+'/data/val'
        test_path = result_path+'/data/test'
        train_path_tmps = preprocessing_path+'/datasplit/train'
        val_path_tmps = preprocessing_path+'/datasplit/val'
        test_path_tmps = preprocessing_path+'/datasplit/test'
        os.makedirs(train_path, exist_ok=True)
        valid_extensions = ['jpg', 'JPG', 'png', 'PNG']
        for (root, directories, files) in os.walk(train_path_tmps):
            for file in files:
                if file.split('.')[-1] in valid_extensions:
                    file_path = os.path.join(root, file)
                    shutil.copy2(file_path, train_path)
        os.makedirs(val_path, exist_ok=True)
        for (root, directories, files) in os.walk(val_path_tmps):
            for file in files:
                if file.split('.')[-1] in valid_extensions:
                    file_path = os.path.join(root, file)
                    shutil.copy2(file_path, val_path)
        os.makedirs(test_path, exist_ok=True)
        for (root, directories, files) in os.walk(test_path_tmps):
            for file in files:
                if file.split('.')[-1] in valid_extensions:
                    file_path = os.path.join(root, file)
                    shutil.copy2(file_path, test_path)
        def pascal_voc_to_yolo_bbox(bbox_array, w, h):
            x_min, y_min, x_max, y_max = bbox_array
            x_center = ((x_max + x_min) / 2) / w
            y_center = ((y_max + y_min) / 2) / h
            width = (x_max - x_min) / w
            height = (y_max - y_min) / h  
            return [x_center, y_center, width, height]
        image_list_train = sorted([f for f in os.listdir(train_path) if f.endswith(tuple(['jpg', 'png', 'JPG', 'PNG']))])
        image_list_val = sorted([f for f in os.listdir(val_path) if f.endswith(tuple(['jpg', 'png', 'JPG', 'PNG']))])
        image_list_test = sorted([f for f in os.listdir(test_path) if f.endswith(tuple(['jpg', 'png', 'JPG', 'PNG']))])
        def create_label_file(image_items, folder_name):
            for image in image_items:
                print(image)
                fileName = Path(image).stem
                print(fileName)
                df = df_data[df_data['filename'] == image]
                print(df)
                with open(folder_name + "/" + fileName +'.txt', 'w') as f:
                    for i in range(0, len(df)):
                        bbox = pascal_voc_to_yolo_bbox(df.iloc[i]['bboxes'], df.iloc[i]['width'], df.iloc[i]['height'])
                        bbox_text = " ".join(map(str, bbox))
                        txt = str(df.iloc[i]['class_id'])+ " " + bbox_text
                        f.write(txt)
                        if i != len(df) - 1:
                            f.write("\n")
        create_label_file(image_list_train, train_path)
        create_label_file(image_list_val, val_path)
        create_label_file(image_list_test, test_path)
        for i in os.listdir(train_path):
            client.fput_object(bucket_name=bucket, object_name=projectId+'/'+versionId+'/train/data/train/'+i, file_path=train_path+'/'+i)
        for i in os.listdir(val_path):
            client.fput_object(bucket_name=bucket, object_name=projectId+'/'+versionId+'/train/data/val/'+i, file_path=val_path+'/'+i)
        for i in os.listdir(test_path):
            client.fput_object(bucket_name=bucket, object_name=projectId+'/'+versionId+'/train/data/test/'+i, file_path=test_path+'/'+i)
        yaml = f"""
            train: {train_path}
            val: {val_path}
            test: {test_path}
            nc: {class_count}
            names: {class_lists}
            """
        with open(result_path+'/custom.yaml', 'w') as f:
            f.write(yaml)
        yaml_file = [file for file in os.listdir(result_path) if file.endswith('.yaml')]
        for i in yaml_file:
            client.fput_object(bucket_name=bucket, object_name=projectId+'/'+versionId+'/train/'+i, file_path=result_path+'/'+i)
        cuda = torch.cuda.is_available()
        print(cuda)

        if cuda == True:
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

            db = pymysql.connect(
                host = '10.40.217.236',
                user = 'root',
                password = 'password',
                port = 3306,
                db = 'yolo',
                charset = 'utf8',
            )

            #db = pymysql.connect(host='10.40.217.236', user='root', password='password', port=3307, db='sms', charset='utf8')

            def db_mysql_training_update(projectId, versionId, trainProgress, epoch):
                cursor = db.cursor()
                try:
                    sql = 'Update Training set trainProgress=%s, epoch=%s where (projectId, versionId)=%s'
                    val = [trainProgress, epoch, (projectId, versionId)]
                    cursor.execute(sql, val)
                    db.commit()
                finally:        
                    cursor.close()        

            class CustomCallback:
                def __init__(self, epoch):
                    self.epoch = epoch
                    self.start_time = datetime.now()
        
                def __call__(self, model, *args, **kwargs):
                    try:
                        current_time = datetime.now().strftime('%H:%M:%S')
                        epoch = getattr(model, 'epoch', 'unknown')
                        if epoch != 'unknown':
                            log_file_path = result_path+'/training_progress.txt'
                            with open(log_file_path, 'w') as f:
                                progress_percentage = (epoch / self.epoch) * 100
                                print(progress_percentage)

                                db_mysql_training_update(
                                    projectId = projectId,
                                    versionId = versionId,
                                    trainProgress = progress_percentage,
                                    epoch = epoch
                                )

                                f.seek(0)
                                f.write(f"Current time: {current_time}\nCurrent progress: {progress_percentage:.2f}%")
                        else:
                            logging.info(f"Epoch information is not available. Progress: unknown% at {current_time}.")
                    except Exception as e:
                        logging.error(f"Error in CustomCallback: {e}")            

            if algorithm == 'yolo_version_5_normal':
                model = YOLO('yolov5nu.pt')
            elif algorithm == 'yolo_version_5_small':
                model = YOLO('yolov5su.pt')
            elif algorithm == 'yolo_version_5_medium':
                model = YOLO('yolov5mu.pt')
            elif algorithm == 'yolo_version_5_large':
                model = YOLO('yolov5lu.pt')
            elif algorithm == 'yolo_version_5_xlarge':
                model = YOLO('yolov5xu.pt')
            elif algorithm == 'yolo_version_8_normal':
                model = YOLO('yolov8n.pt')
            elif algorithm == 'yolo_version_8_small':
                model = YOLO('yolov8s.pt')
            elif algorithm == 'yolo_version_8_medium':
                model = YOLO('yolov8m.pt')
            elif algorithm == 'yolo_version_8_large':
                model = YOLO('yolov8l.pt')
            elif algorithm == 'yolo_version_8_xlarge':
                model = YOLO('yolov8x.pt')

            ### new 2025.01.03 ################################################

            elif algorithm == 'yolo_version_6_normal':
                model = YOLO('yolov6n.yaml')
            elif algorithm == 'yolo_version_6_small':
                model = YOLO('yolov6s.yaml')
            elif algorithm == 'yolo_version_6_medium':
                model = YOLO('yolov6m.yaml')
            elif algorithm == 'yolo_version_6_large':
                model = YOLO('yolov6l.yaml')
            elif algorithm == 'yolo_version_6_xlarge':
                model = YOLO('yolov6x.yaml')

            elif algorithm == 'yolo_version_8_normal':
                model = YOLO('yolov8n.pt')
            elif algorithm == 'yolo_version_8_small':
                model = YOLO('yolov8s.pt')
            elif algorithm == 'yolo_version_8_medium':
                model = YOLO('yolov8m.pt')
            elif algorithm == 'yolo_version_8_large':
                model = YOLO('yolov8l.pt')
            elif algorithm == 'yolo_version_8_xlarge':
                model = YOLO('yolov8x.pt')
            elif algorithm == 'yolo_version_9_normal':
                model = YOLO('yolov9t.pt')
            elif algorithm == 'yolo_version_9_small':
                model = YOLO('yolov9s.pt')
            elif algorithm == 'yolo_version_9_medium':
                model = YOLO('yolov9m.pt')
            elif algorithm == 'yolo_version_9_large':
                model = YOLO('yolov9c.pt')
            elif algorithm == 'yolo_version_9_xlarge':
                model = YOLO('yolov9e.pt')
            elif algorithm == 'yolo_version_10_normal':
                model = YOLO('yolov10n.pt')
            elif algorithm == 'yolo_version_10_small':
                model = YOLO('yolov10s.pt')
            elif algorithm == 'yolo_version_10_medium':
                model = YOLO('yolov10m.pt')
            elif algorithm == 'yolo_version_10_large':
                model = YOLO('yolov10l.pt')
            elif algorithm == 'yolo_version_10_xlarge':
                model = YOLO('yolov10x.pt')
            elif algorithm == 'yolo_version_11_normal':
                model = YOLO('yolo11n.pt')
            elif algorithm == 'yolo_version_11_small':
                model = YOLO('yolo11s.pt')
            elif algorithm == 'yolo_version_11_medium':
                model = YOLO('yolo11m.pt')
            elif algorithm == 'yolo_version_11_large':
                model = YOLO('yolo11l.pt')
            elif algorithm == 'yolo_version_11_xlarge':
                model = YOLO('yolo11x.pt')

            ###################################################################

            custom_callback = CustomCallback(epoch=epoch)

            model.add_callback('on_train_epoch_end', custom_callback)

            model.train(
                data = result_path+'/custom.yaml',
                batch = batchsize,
                epochs = epoch,
                project = result_path+'/model',
                exist_ok = True,
                device = 1 
            )
            score = model.val(data = result_path+'/custom.yaml')

            log_file_path = result_path+'/training_progress.txt'
            with open(log_file_path, 'w') as f:
                current_time = datetime.now().strftime('%H:%M:%S')
                progress_percentage = 100  
                f.seek(0)
                f.write(f"Current time: {current_time}\nCurrent progress: {progress_percentage:.2f}%")

            values = []
            for i in list(score.results_dict.values())[0:3]:
                value = '{:.4f}'.format(i)
                values.append(value)
            values_np = np.column_stack((['precision', 'recall', 'mAP'], values))
            values_pd = pd.DataFrame(values_np)
            values_pd.to_csv(result_path+'/metrics_score.csv')
            metrics_score = [file for file in os.listdir(result_path) if file.endswith('metrics_score.csv')]
            for i in metrics_score:
                client.fput_object(bucket_name=bucket, object_name=projectId+'/'+versionId+'/train/'+i, file_path=result_path+'/'+i)
            best_pt = [file for file in os.listdir(result_path+'/model/train/weights') if file.endswith('best.pt')]
            for i in best_pt:
                client.fput_object(bucket_name=bucket, object_name=projectId+'/'+versionId+'/train/model/train/weight/'+i, file_path=result_path+'/model/train/weights/'+i)

            ### 모델 다운로드

            def install_p7zip():
                try:
                    subprocess.run(['apt-get', 'install', '-y', 'p7zip-full'], check=True)
                    print("p7zip-full이 성공적으로 설치되었습니다.")
                except subprocess.CalledProcessError as e:
                    print(f"설치 중 오류가 발생했습니다: {e}")

            install_p7zip()

            #def create_split_zip(input_path, output_path, zip_name="model_weight", split_size="1024M"):
            #    files = [os.path.join(input_path, f) for f in os.listdir(input_path) if os.path.isfile(os.path.join(input_path, f))]
            #    zip_file_base = os.path.join(output_path, zip_name)
            #    command = ["7z", "a", f"{zip_file_base}.7z", *files, f"-v{split_size}"]
            #    subprocess.run(command, check=True)
            #    print(f"분할 압축이 완료되었습니다: {zip_file_base}.7z")

            def create_zip(input_path, output_path, zip_name="model_weight", max_size=10, split_size="2M"):
                files = []
                for root, dirs, file_names in os.walk(input_path):
                    for file_name in file_names:
                        file_path = os.path.join(root, file_name)
                        files.append(file_path)
                zip_file_base = os.path.join(output_path, zip_name)
                total_size = sum(os.path.getsize(f) for f in files)
                total_size_MB = total_size / (1024 * 1024)
                if total_size_MB <= max_size:
                    command = ["7z", "a", f"{zip_file_base}.7z", *files]
                    subprocess.run(command, check=True)
                    print(f"압축이 완료되었습니다: {zip_file_base}.7z")
                else:
                    command = ["7z", "a", f"{zip_file_base}.7z", *files, f"-v{split_size}"]
                    subprocess.run(command, check=True)
                    print(f"분할 압축이 완료되었습니다: {zip_file_base}.7z")

            input_path = result_path+'/model/train/weights'
            output_path = result_path+'/model/train/weights_zip'
            if not os.path.exists(output_path):
                os.makedirs(output_path)

            create_zip(input_path, output_path, zip_name="model_weight", max_size=1024, split_size="1024M")

            zip_files = [file for file in os.listdir(output_path) if file.startswith('model_weight')]
            for i in zip_files:
                client.fput_object(bucket_name=bucket, object_name=projectId+'/'+versionId+'/train/model_weight/'+i, file_path=output_path+'/'+i)

            ########################################################################

            df = pd.read_csv(result_path+'/model/train/results.csv', header=0)
            df.columns = df.columns.str.strip()
            fig, ax = plt.subplots(figsize=(16, 8), facecolor='white')
            ax.plot(df['train/box_loss'].values, '.-', label='train', color='b')
            ax.plot(df['val/box_loss'].values, '.-', label='validation', color='r')
            ax.set(
                xlabel='Epoch',
                ylabel='Box Loss'
            )
            ax.grid()
            ax.patch.set_facecolor('white')
            ax.legend(loc=0)
            plt.axvline(x=0, color = 'black')
            plt.axhline(y=0, color = 'black')
            plt.savefig(result_path+'/model/train/metrics_chart.jpg', transparent = True)
            metrics_chart = [file for file in os.listdir(result_path+'/model/train') if file.endswith('metrics_chart.jpg')]
            for i in metrics_chart:
                client.fput_object(bucket_name=bucket, object_name=projectId+'/'+versionId+'/train/'+i, file_path=result_path+'/model/train/'+i)
            validation_result = model.predict(source=val_path, save=True, project=result_path, name='result', exist_ok=True)
#            for i in os.listdir(result_path+'/result'):
#                client.fput_object(bucket_name=bucket, object_name=projectId+'/'+versionId+'/train/result/'+i, file_path=result_path+'/result/'+i)
            sample(result_path+'/result')
            validation_results = [file for file in os.listdir(result_path) if file.endswith('validation_result.jpg')]
            for i in validation_results:
                client.fput_object(bucket_name=bucket, object_name=projectId+'/'+versionId+'/train/'+i, file_path=result_path+'/'+i)

            print('Training Done')

            db_mysql_training_update(
                projectId = projectId,
                versionId = versionId,
                trainProgress = 100,
                epoch = epoch
            )

        else:
            print('GPU is not using')

    ################################################################################################
    ## preprocessing task 1 run
    ################################################################################################

    def db_mysql_stat_update(projectId, versionId, statusOfTrain):
        cursor = db.cursor()
        try:
            sql = 'Update Stat set statusOfTrain=%s where (projectId, versionId)=%s'
            val = [statusOfTrain, (projectId, versionId)]
            cursor.execute(sql, val)
            db.commit()
        finally:        
            cursor.close()        

    def db_mysql_training_update(projectId, versionId, algorithm, batchsize, mAP, recall, precisions, subStatusOfTraining):
        cursor = db.cursor()
        try:
            sql = 'Update Training set algorithm=%s, batchsize=%s, mAP=%s, recall=%s, precisions=%s, subStatusOfTraining=%s where (projectId, versionId)=%s'
            val = [algorithm, batchsize, mAP, recall, precisions, subStatusOfTraining, (projectId, versionId)]
            cursor.execute(sql, val)
            db.commit()
        finally:        
            cursor.close()        

    def db_mysql_training_tmp_update(projectId, versionId, subStatusOfTraining):
        cursor = db.cursor()
        try:
            sql = 'Update Training set subStatusOfTraining=%s where (projectId, versionId)=%s'
            val = [subStatusOfTraining, (projectId, versionId)]
            cursor.execute(sql, val)
            db.commit()
        finally:        
            cursor.close()        
        

    def db_mysql_training_error_update(projectId, versionId, TrainingErrorCategory, TrainingErrorLog):
        cursor = db.cursor()
        try:
            sql = 'Update Error set TrainingErrorCategory=%s, TrainingErrorLog=%s where (projectId, versionId)=%s'
            val = [TrainingErrorCategory, TrainingErrorLog, (projectId, versionId)]
            cursor.execute(sql, val)
            db.commit()
        finally:        
            cursor.close()        
        
    db = pymysql.connect(host = '10.40.217.236', user = 'root', password = 'password', port = 3306, db = 'yolo', charset = 'utf8')

    try:
        if algorithm in ('yolo_version_5_normal', 'yolo_version_5_small', 'yolo_version_5_medium', 'yolo_version_5_large', 'yolo_version_5_xlarge', 'yolo_version_8_normal', 'yolo_version_8_small', 'yolo_version_8_medium', 'yolo_version_8_large', 'yolo_version_8_xlarge'):
            db_mysql_stat_update(projectId = projectId, versionId = versionId, statusOfTrain = 'RUNNING')
            db_mysql_training_tmp_update(projectId = projectId, versionId = versionId, subStatusOfTraining = 'RUNNING')
            db_mysql_training_error_update(projectId = projectId, versionId = versionId, TrainingErrorCategory = 0, TrainingErrorLog = '정상')
            yolo()

    except Exception as e:
        db_mysql_stat_update(projectId = projectId, versionId = versionId, statusOfTrain = 'ERROR')
        db_mysql_training_tmp_update(projectId = projectId, versionId = versionId, subStatusOfTraining = 'ERROR')

        error_message = f"Error: {e}\n"

        with open(result_path+"/error.log", "w") as f:
            f.write(error_message)

        #error_log = [file for file in os.listdir(result_path) if file.endswith('error.log')]
        #for i in error_log:
        #    client.fput_object(bucket_name=bucket, object_name=projectId+'/'+versionId+'/train/'+i, file_path=result_path+'/'+i)

        if os.path.exists(result_path+"/error.log"):
            with open(result_path+"/error.log", "r") as f:
                error_content = f.read()

            if "CUDA out of memory" in error_content:
                trainingerrorcategory = 1
                #new_content = "GPU 메모리 부족으로 학습이 중단되면서 오류가 발생하였습니다."
            elif "root" in error_content or "broken permissions" in error_content:
                trainingerrorcategory = 2
                #new_content = "컨테이너 연결 과정에서 root 권한이 부여되지 않아 오류가 발생하였습니다."
            elif "BadRequest" in error_content or "PodInitializing" in error_content:
                trainingerrorcategory = 3
                #new_content = "컨테이너를 생성하지 못하여 오류가 발생하였습니다."
                #new_content = "알 수 없는 오류"

            #with open(result_path+"/error_message.txt", "w") as f:
            #    f.write(new_content)

        db_mysql_training_error_update(projectId = projectId, versionId = versionId, TrainingErrorCategory = trainingerrorcategory, TrainingErrorLog = error_message)

        print(f"Error: {e}")

        sys.exit(1)
    
    else:
        try:
            metrics_file = f"{projectId}/{versionId}/train/metrics_score.csv"
            for item in client.list_objects(bucket_name=bucket, prefix=f"{projectId}/{versionId}/train", recursive=True):
                if item.object_name == metrics_file:
                    metrics = client.get_object(bucket, item.object_name)
                    df_metrics = pd.read_csv(metrics, index_col=0, names=['metrics', 'value'], header=0)
                    break
            else:
                print("Metrics file not found")
                return    

            precision = df_metrics['value'][0]
            recall = df_metrics['value'][1]
            mAP = df_metrics['value'][2]

            db_mysql_training_update(projectId = projectId, versionId = versionId, algorithm = algorithm, batchsize = batchsize, mAP = mAP, recall = recall, precisions = precision, subStatusOfTraining = 'FINISH')
        except Exception as e:
            print(f"Error while processing metrics: {e}")
            sys.exit(1)

################################################################################################
## kubeflow pipeline upload
################################################################################################

def pipelines():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--projectId', type=str)
    parser.add_argument('--versionId', type=str)
    parser.add_argument('--algorithm', type=str)
    parser.add_argument('--batchsize', type=int)
    parser.add_argument('--epoch', type=int)
    args = parser.parse_args()

    Training_apply = Training(args.projectId, args.versionId, args.algorithm, args.batchsize, args.epoch) \
        .set_display_name('Model Training') \
        .apply(onprem.mount_pvc('dlabflow-claim', volume_name='data', volume_mount_path='/mnt/dlabflow')) \
        .add_env_variable(V1EnvVar(name="CUDA_VISIBLE_DEVICES", value="GPU-68f953d4-c0e6-35ce-2435-97b0f64c1bbc"))
    
    smh_vol = kfp.dsl.PipelineVolume(name = 'shm-vol', empty_dir = {'medium': 'Memory'})
    Training_apply.add_pvolumes({'/dev/shm': smh_vol})        
    Training_apply.execution_options.caching_strategy.max_cache_staleness = 'P0D'

if __name__ == '__main__':
    pipeline_package_path = 'training_pipelines.zip'
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
    experiment = client.create_experiment(name='Training')
    run = client.run_pipeline(experiment.id, 'Training pipelines', pipeline_package_path)

