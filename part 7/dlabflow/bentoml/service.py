import subprocess
import sys
import os
import io
import uuid
import ast
from pydantic import BaseModel
import typing as t
import bentoml
from bentoml.io import JSON, File
from bentoml.exceptions import BentoMLException
import pandas as pd
import xml.etree.ElementTree as ET
import shutil
from minio import Minio
import pymysql
import warnings
from multiprocessing import Process

warnings.filterwarnings('ignore')

svc = bentoml.Service('kubeflow')

class DataSourceParams(BaseModel):
    projectId: str
    versionId: str    
    folder: str
    filename: str
    path: str
    database: str
    width: int
    height: int
    depth: int
    segmented: int
    name: str
    obj_id: int
    pose: str
    truncated: int
    difficult: int
    xmin: int
    ymin: int
    xmax: int
    ymax: int
    totalPages: int

class PreprocessingParams(BaseModel):
    projectId: str
    versionId: str
    dataPath: t.List[str]
    dataNormalization: t.List[str]
    dataAugmentation: t.List[str]
    trainRatio: int
    validationRatio: int
    testRatio: int

class TrainingParams(BaseModel):
    projectId: str
    versionId: str
    algorithm: str
    batchsize: int
    epoch: int

class InferenceParams(BaseModel):
    projectId: str
    versionId: str
    sessionId: str

input_spec_datasource = JSON(pydantic_model=DataSourceParams)

# 백그라운드 작업을 위한 함수 정의
def background_datasource_task(arg_dict, projectId, versionId):
    bucket_name = 'aiproject'
    base_path = '/mnt/dlabflow/backend/minio/aiproject'
    annotations_paths = base_path + '/' + projectId + '/rawdata/annotations'
    images_paths = base_path + '/' + projectId + '/rawdata/images'
    os.makedirs(annotations_paths, exist_ok=True)
    os.makedirs(images_paths, exist_ok=True)

    def indent(elem, level=0):
        i = '\n' + level * '  '
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = i + '  '
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
            for elem in elem:
                indent(elem, level + 1)
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = i

    def duplicated(idx, df):
        root = ET.Element('annotation')
        folder = ET.SubElement(root, 'folder').text = df['folder'][idx]
        filename = ET.SubElement(root, 'filename').text = df['filename'][idx]
        path = ET.SubElement(root, 'path').text = df['path'][idx]
        source = ET.SubElement(root, 'source')
        database = ET.SubElement(source, 'database').text = 'GRIT-Dlabflow'
        size = ET.SubElement(root, 'size')
        width = ET.SubElement(size, 'width').text = str(df['width'][idx])
        height = ET.SubElement(size, 'height').text = str(df['height'][idx])
        depth = ET.SubElement(size, 'depth').text = str(df['depth'][idx])
        segmented = ET.SubElement(root, 'segmented').text = str(df['segmented'][idx])
        def add_objects(name, pose, truncated, difficult, obj_id, xmin, xmax, ymin, ymax):
            objects = ET.SubElement(root, 'object')
            ET.SubElement(objects, 'name').text = name
            ET.SubElement(objects, 'pose').text = pose
            ET.SubElement(objects, 'truncated').text = truncated
            ET.SubElement(objects, 'difficult').text = difficult
            ET.SubElement(objects, 'occluded').text = obj_id
            bndbox = ET.SubElement(objects, 'bndbox')
            ET.SubElement(bndbox, 'xmin').text = xmin
            ET.SubElement(bndbox, 'xmax').text = xmax
            ET.SubElement(bndbox, 'ymin').text = ymin
            ET.SubElement(bndbox, 'ymax').text = ymax
        duplicated_rows = df[df['filename'] == df['filename'][idx]]
        for index, obj_data in duplicated_rows.iterrows():
            add_objects(obj_data['name'], obj_data['pose'], str(obj_data['truncated']), str(obj_data['difficult']), str(obj_data['obj_id']), str(obj_data['xmin']), str(obj_data['xmax']), str(obj_data['ymin']), str(obj_data['ymax']))
        indent(root)
        tree = ET.ElementTree(root)
        tree.write(annotations_paths + '/' + os.path.splitext(df['filename'][idx])[0] + '.xml')

    def db_mysql_dataframe(sql_select):
        db = pymysql.connect(host='10.40.217.236', user='root', password='password', port=3306, db='yolo', charset='utf8')
        cursor = db.cursor()
        try:
            sql = f"select * from {sql_select}"
            cursor.execute(sql)
            db.commit()
            df = pd.read_sql(sql, db)
        finally:
            cursor.close()
        return df

    def db_mysql_stat_update(sql_select, projectId, versionId, statusOfDataSource):
        db = pymysql.connect(host='10.40.217.236', user='root', password='password', port=3306, db='yolo', charset='utf8')
        cursor = db.cursor()
        try:
            sql = f"Update {sql_select} set statusOfDataSource=%s where (projectId, versionId)=%s"
            val = [statusOfDataSource, (projectId, versionId)]
            cursor.execute(sql, val)
            db.commit()
        finally:
            cursor.close()        

    def db_mysql_preprocessing_update(sql_select, projectId, versionId, PageN, Pagelast):
        db = pymysql.connect(host='10.40.217.236', user='root', password='password', port=3306, db='yolo', charset='utf8')
        cursor = db.cursor()
        try:
            sql = f"UPDATE {sql_select} SET PageN=%s, Pagelast=%s WHERE (projectId, versionId)=%s"
            val = [PageN, Pagelast, (projectId, versionId)]
            cursor.execute(sql, val)
            db.commit()
        finally:
            cursor.close()

    df_mysql_1 = db_mysql_dataframe(sql_select='Stat')
    df_mysql_select_1 = df_mysql_1[(df_mysql_1['projectId'] == projectId) & (df_mysql_1['versionId'] == versionId)]
    df_mysql_2 = db_mysql_dataframe(sql_select='Preprocessing')
    df_mysql_select_2 = df_mysql_2[(df_mysql_2['projectId'] == projectId) & (df_mysql_2['versionId'] == versionId)]

    values = [DataSourceParams(**item).dict() for item in arg_dict]
    df = pd.DataFrame(values)

    totalpages = df['totalPages'].values[0]

    # If totalpages is greater than 1
    if totalpages > 1:
        if 'READY' in df_mysql_select_1['statusOfDataSource'].values:
            status_data_source = 'RUNNING'
            page = 0
            pagen = 1
        else:
            status_data_source = 'RUNNING'
            page = df_mysql_select_2['PageN'].values[0]
            pagen = page + 1
            if pagen >= totalpages - 1:
                pagen = totalpages - 1
                status_data_source = 'FINISH'

        print('totalpages is greater than 1')
        print(f"totalPages: {totalpages}, page: {page}, pagen: {pagen}")

    # If totalpages is 1
    else:
        status_data_source = 'FINISH'
        pagen = 1

        print('totalpages is 1')
        print(f"totalPages: {totalpages}, pagen: {pagen}, status: {status_data_source}")    

    if status_data_source == 'FINISH':
        try:
            object_name = f'{projectId}_{versionId}.csv'
            client = Minio(endpoint='10.40.217.236:9002', access_key='dlab-backend', secret_key='dlab-backend-secret', secure=False)

            print('MinIO Check')

            response = client.get_object(bucket_name, object_name)
            csv_data = response.read().decode('utf-8')
            response.close()
            response.release_conn()
            minio_df = pd.read_csv(io.StringIO(csv_data))
            projectid = minio_df['projectId'].values[0]
            versionid = minio_df['versionId'].values[0]
            datapath = minio_df['dataPath'].values[0]
            datanormalization = minio_df['dataNormalization'].values[0]
            dataaugmentation = minio_df['dataAugmentation'].values[0]
            trainratio = minio_df['trainRatio'].values[0]
            validationratio = minio_df['validationRatio'].values[0]
            testratio = minio_df['testRatio'].values[0]

            os.system('python3 /mnt/dlabflow/backend/kubeflow/pipelines/admin/preprocessing.py --projectId=%s --versionId=%s --dataPath=%s --dataNormalization=%s --dataAugmentation=%s --trainRatio=%s --validationRatio=%s --testRatio=%s' % (projectid, versionid, datapath, datanormalization, dataaugmentation, trainratio, validationratio, testratio))
        except Exception as e:
            print(f"Error before MinIO Check: {e}")

    try:
        annotation_list = [os.path.join(root, file) for root, _, files in os.walk(annotations_paths) for file in files if file.endswith('xml')]
        if not os.listdir(annotations_paths):
            for idx, _ in df.iterrows():
                duplicated(idx, df)
                db_mysql_stat_update(sql_select='Stat', projectId=projectId, versionId=versionId, statusOfDataSource=status_data_source)
                db_mysql_preprocessing_update(sql_select='Preprocessing', projectId=projectId, versionId=versionId, PageN=pagen, Pagelast=totalpages-1)
        else:
            for idx in range(len(df)):
                duplicated(idx, df)
                db_mysql_stat_update(sql_select='Stat', projectId=projectId, versionId=versionId, statusOfDataSource=status_data_source)
                db_mysql_preprocessing_update(sql_select='Preprocessing', projectId=projectId, versionId=versionId, PageN=pagen, Pagelast=totalpages-1)

        client = Minio(endpoint='10.40.217.236:9002', access_key='dlab-backend', secret_key='dlab-backend-secret', secure=False)
        bucket = df['folder'][0].split('/')[0]
        prefix = df['folder'][0].split('/')[1]
        objects = client.list_objects(bucket, prefix+'/')
        for obj in objects:
            filename = os.path.basename(obj.object_name)
            for i in range(len(df)):
                if filename == df['filename'][i]:
                    client.fget_object(bucket_name=bucket, object_name=obj.object_name, file_path=images_paths+'/'+filename)
        for i in os.listdir(annotations_paths):
            client.fput_object(bucket_name=bucket_name, object_name=projectId+'/rawdata/annotations/'+i, file_path=annotations_paths+'/'+i)
        for i in os.listdir(images_paths):
            client.fput_object(bucket_name=bucket_name, object_name=projectId+'/rawdata/images/'+i, file_path=images_paths+'/'+i)
    except Exception as e:
        print(f"Error in background task: {e}")
        status_data_source = 'ERROR'
        db_mysql_stat_update(sql_select='Stat', projectId=projectId, versionId=versionId, statusOfDataSource=status_data_source)

@svc.api(input=JSON(), output=JSON(), route='/datasource')
def datasource(arg: PreprocessingParams):
    print('+-'*20 + '+')
    print('datasource')
    print(arg)

    projectId = arg[0]['projectId']
    versionId = arg[0]['versionId']

    # 백그라운드 프로세스 시작
    p = Process(target=background_datasource_task, args=(arg, projectId, versionId))
    p.start()

    # 즉시 응답 반환
    return {"status": "accepted", "message": "Task is being processed in the background", "projectId": projectId, "versionId": versionId}

input_spec_preprocessing = JSON(pydantic_model=PreprocessingParams)

# 백그라운드 작업을 위한 함수 정의
def background_preprocessing_task(arg_dict):
    base_path = '/mnt/dlabflow/backend/minio/aiproject'
    os.makedirs(base_path, exist_ok=True)
    
    arg = PreprocessingParams(**arg_dict)
    arg_dataPath = [", ".join(arg.dataPath)]
    arg_dataNormalization = [", ".join(arg.dataNormalization)]
    arg_dataAugmentation = [", ".join(arg.dataAugmentation)]

    def db_mysql_dataframe(sql_select):
        try:
            allowed_tables = {"Stat", "Preprocessing"}
            if sql_select not in allowed_tables:
                raise ValueError(f"Invalid table name: {sql_select}")
            db = pymysql.connect(host='10.40.217.236', user='root', password='password', port=3306, db='yolo', charset='utf8')
            sql = f"SELECT * FROM {sql_select}"
            df = pd.read_sql(sql, db)
            return df
        except Exception as e:
            print(f"Error: {e}")
            return None
        finally:
            db.close()

    def db_mysql_stat_update(sql_select, projectId, versionId, statusOfProject):
        try:
            allowed_tables = {"Stat", "Preprocessing"}
            if sql_select not in allowed_tables:
                raise ValueError(f"Invalid table name: {sql_select}")
            db = pymysql.connect(host='10.40.217.236', user='root', password='password', port=3306, db='yolo', charset='utf8')
            cursor = db.cursor()
            sql = f"UPDATE {sql_select} SET statusOfProject=%s WHERE (projectId, versionId)=%s"
            val = [statusOfProject, (projectId, versionId)]
            cursor.execute(sql, val)
            db.commit()
        except Exception as e:
            print(f"Error: {e}")
        finally:
            cursor.close()
            db.close()

    client = Minio(endpoint='10.40.217.236:9002', access_key='dlab-backend', secret_key='dlab-backend-secret', secure=False)
    df = pd.DataFrame({'projectId': [arg.projectId], 'versionId': [arg.versionId], 'dataPath': [arg_dataPath], 'dataNormalization': [arg_dataNormalization], 'dataAugmentation': [arg_dataAugmentation], 'trainRatio': [arg.trainRatio], 'validationRatio': [arg.validationRatio], 'testRatio': [arg.testRatio]})

    project_name = df.loc[0, 'projectId']
    version_name = df.loc[0, 'versionId']
    bucket_name = 'aiproject'
    object_name = f'{project_name}_{version_name}.csv'

    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_bytes = csv_buffer.getvalue().encode('utf-8')
    client.put_object(bucket_name=bucket_name, object_name=object_name, data=io.BytesIO(csv_bytes), length=len(csv_bytes), content_type='application/csv')

    df_mysql = db_mysql_dataframe(sql_select='Stat')
    df_mysql_select = df_mysql[(df_mysql['projectId'] == arg.projectId) & (df_mysql['versionId'] == arg.versionId)]

    try:
        if 'READY' in df_mysql_select['statusOfProject'].values:
            status_project = 'FINISH'
            db_mysql_stat_update(sql_select='Stat', projectId=arg.projectId, versionId=arg.versionId, statusOfProject=status_project)
    except Exception as e:
        print(f"Error in background task: {e}")
        status_project = 'Error'
        db_mysql_stat_update(sql_select='Stat', projectId=arg.projectId, versionId=arg.versionId, statusOfProject=status_project)

@svc.api(input=input_spec_preprocessing, output=JSON(), route='/preprocessing')
def preprocessing(arg: PreprocessingParams):
    print('preprocessing')
    print(arg)

    # 백그라운드 프로세스 시작
    p = Process(target=background_preprocessing_task, args=(arg.dict(),))
    p.start()

    # 즉시 응답 반환
    return {"status": "accepted", "message": "Preprocessing task is being processed in the background", "projectId": arg.projectId, "versionId": arg.versionId}

input_spec_training = JSON(pydantic_model=TrainingParams)

# 백그라운드 작업을 위한 함수 정의
def background_training_task(arg_dict):
    arg = TrainingParams(**arg_dict)
    if arg.algorithm in ('yolo_version_5_normal', 'yolo_version_5_small', 'yolo_version_5_medium', 'yolo_version_5_large', 'yolo_version_5_xlarge', 'yolo_version_8_normal', 'yolo_version_8_small', 'yolo_version_8_medium', 'yolo_version_8_large', 'yolo_version_8_xlarge'):
        os.system('python3 /mnt/dlabflow/backend/kubeflow/pipelines/admin/training_yolo.py --projectId=%s --versionId=%s --algorithm=%s --batchsize=%s --epoch=%s' % (arg.projectId, arg.versionId, arg.algorithm, arg.batchsize, arg.epoch))
    elif arg.algorithm in ('efficientdet_d0', 'efficientdet_d1', 'efficientdet_d2', 'efficientdet_d3', 'efficientdet_d4', 'efficientdet_d5', 'efficientdet_d6', 'efficientdet_d7'):
        os.system('python3 /mnt/dlabflow/backend/kubeflow/pipelines/admin/training_tf.py --projectId=%s --versionId=%s --algorithm=%s --batchsize=%s --epoch=%s' % (arg.projectId, arg.versionId, arg.algorithm, arg.batchsize, arg.epoch))        

@svc.api(input=input_spec_training, output=JSON(), route='/training')
def training(arg: TrainingParams):
    print('training')
    print(arg)

    # 백그라운드 프로세스 시작
    p = Process(target=background_training_task, args=(arg.dict(),))
    p.start()

    # 즉시 응답 반환
    return {"status": "accepted", "message": "Training task is being processed in the background", "projectId": arg.projectId, "versionId": arg.versionId}

input_spec_inference = JSON(pydantic_model=InferenceParams)

# 백그라운드 작업을 위한 함수 정의
def background_inference_task(arg_dict):
    arg = InferenceParams(**arg_dict)
    os.system('python3 /mnt/dlabflow/backend/kubeflow/pipelines/admin/inference_yolo.py --projectId=%s --versionId=%s --sessionId=%s' % (arg.projectId, arg.versionId, arg.sessionId))

    cmd = [
        'python3',
        '/mnt/dlabflow/backend/kubeflow/pipelines/admin/inference_yolo.py',
        '--projectId', arg.projectId,
        '--versionId', arg.versionId,
        '--sessionId', arg.sessionId
    ]
    #subprocess.Popen(cmd)  # 비동기 실행    

@svc.api(input=input_spec_inference, output=JSON(), route='/inference')
def inference(arg: InferenceParams):
    print('inference')
    print(arg)

    # 백그라운드 프로세스 시작
    p = Process(target=background_inference_task, args=(arg.dict(),))
    p.start()

    # 즉시 응답 반환
    return {"status": "accepted", "message": "Inference task is being processed in the background", "projectId": arg.projectId, "versionId": arg.versionId}

