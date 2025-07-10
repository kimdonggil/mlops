import subprocess
import sys

#try:
#    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pydantic', 'minio', 'PyMySQL'])
#except subprocess.CalledProcessError as e:
#    print(f"Error occurred while installing packages: {e}")
#    sys.exit(1)

try:
    import os
    import io
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
    warnings.filterwarnings('ignore')
except ImportError as e:
    print(f"Error occurred while importing modules: {e}")
    sys.exit(1)

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
    inferenceAlgorithm: str

input_spec_datasource = JSON(pydantic_model=DataSourceParams)

@svc.api(input=JSON(), output=JSON(), route='/datasource')
def datasource(arg: PreprocessingParams):
    print('datasource')
    print(arg)
    print(type(arg))
    print(len(arg))

    bucket_name = 'aiproject'

    projectId = arg[0]['projectId']
    versionId = arg[0]['versionId']

    base_path = '/mnt/dlabflow/backend/minio/aiproject'
    annotations_paths = base_path+'/'+projectId+'/rawdata/annotations'
    os.makedirs(annotations_paths, exist_ok = True)
    images_paths = base_path+'/'+projectId+'/rawdata/images'
    os.makedirs(images_paths, exist_ok = True)    

    def indent(elem, level = 0):
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
        tree.write(annotations_paths+'/'+os.path.splitext(df['filename'][idx])[0]+'.xml')

    def db_mysql_dataframe(sql_select):
        db = pymysql.connect(host='10.40.217.236', user='root', password='password', port=3306, db='yolo', charset='utf8')
        cursor = db.cursor()
        sql = f"select * from {sql_select}"
        cursor.execute(sql)
        db.commit()
        df = pd.read_sql(sql, db)
        return df

    def db_mysql_stat_update(sql_select, projectId, versionId, statusOfDataSource):
        db = pymysql.connect(host='10.40.217.236', user='root', password='password', port=3306, db='yolo', charset='utf8')
        cursor = db.cursor()
        sql = f"Update {sql_select} set statusOfDataSource=%s where (projectId, versionId)=%s"
        val = [statusOfDataSource, (projectId, versionId)]
        cursor.execute(sql, val)
        db.commit()
        cursor.close()

    def db_mysql_preprocessing_update(sql_select, projectId, versionId, PageN, Pagelast):
        db = pymysql.connect(host='10.40.217.236', user='root', password='password', port=3306, db='yolo', charset='utf8')
        cursor = db.cursor()
        sql = f"Update {sql_select} set PageN=%s, Pagelast=%s where (projectId, versionId)=%s"
        val = [PageN, Pagelast, (projectId, versionId)]
        cursor.execute(sql, val)
        db.commit()
        cursor.close()

    df_mysql_1 = db_mysql_dataframe(sql_select='Stat')
    df_mysql_select_1 = df_mysql_1[(df_mysql_1['projectId'] == projectId) & (df_mysql_1['versionId'] == versionId)]
    df_mysql_2 = db_mysql_dataframe(sql_select='Preprocessing')
    df_mysql_select_2 = df_mysql_2[(df_mysql_2['projectId'] == projectId) & (df_mysql_2['versionId'] == versionId)]  

    values = []
    for i in range(len(arg)):
        data_source = DataSourceParams(projectId=arg[i]['projectId'], versionId=arg[i]['versionId'], folder=arg[i]['folder'], filename=arg[i]['filename'], path=arg[i]['path'], database=arg[i]['database'], width=arg[i]['width'], height=arg[i]['height'], depth=arg[i]['depth'], segmented=arg[i]['segmented'], name=arg[i]['name'], obj_id=arg[i]['obj_id'], pose=arg[i]['pose'], truncated=arg[i]['truncated'], difficult=arg[i]['difficult'], xmin=arg[i]['xmin'], ymin=arg[i]['ymin'], xmax=arg[i]['xmax'], ymax=arg[i]['ymax'], totalPages=arg[i]['totalPages'])
        values.append(data_source.dict())
    df = pd.DataFrame(values)

    totalpages = df['totalPages'].values[0]

    if 'Ready' in df_mysql_select_1['statusOfDataSource'].values:
        status_data_source = 'Running'
        page = 0
        print(f"page{page}")
        pagen = 1
    else:
        status_data_source = 'Running'
        page = df_mysql_select_2['PageN'].values[0]
        print(f"page{page}")
        pagen = page+1
        if pagen > totalpages-1:
            pagen = totalpages-1
            status_data_source = 'Finish'
            object_name = f'{projectId}_{versionId}.csv'           
            client = Minio(endpoint='10.40.217.236:9002', access_key='dlab-backend', secret_key='dlab-backend-secret', secure=False)
            response = client.get_object(bucket_name, object_name)
            csv_data = response.read().decode('utf-8')
            response.close()
            response.release_conn()
            minio_df = pd.read_csv(io.StringIO(csv_data))
            print(minio_df)

            projectid = minio_df['projectId'].values[0]
            versionid = minio_df['versionId'].values[0]
            datapath = minio_df['dataPath'].values[0]
            datanormalization = minio_df['dataNormalization'].values[0]
            dataaugmentation = minio_df['dataAugmentation'].values[0]
            trainratio = minio_df['trainRatio'].values[0]
            validationratio = minio_df['validationRatio'].values[0]
            testratio = minio_df['testRatio'].values[0]          

            os.system('python3 /mnt/dlabflow/backend/kubeflow/pipelines/preprocessing.py --projectId=%s --versionId=%s --dataPath=%s --dataNormalization=%s --dataAugmentation=%s --trainRatio=%s --validationRatio=%s --testRatio=%s' %(projectid, versionid, datapath, datanormalization, dataaugmentation, trainratio, validationratio, testratio))

    try:
        annotation_list = []
        for (root, directories, files) in os.walk(annotations_paths):
            for file in files:
                if file.endswith('xml'):
                    annotation = os.path.join(root, file)
                    annotation_list.append(annotation)
        if not os.listdir(annotations_paths):
            for idx, names in df.iterrows():
                duplicated(idx, df)
                db_mysql_stat_update(sql_select='Stat', projectId=projectId, versionId=versionId, statusOfDataSource=status_data_source)
                db_mysql_preprocessing_update(sql_select='Preprocessing', projectId=projectId, versionId=versionId, PageN=pagen, Pagelast=totalpages-1)

        else:
            for count, annotation in enumerate(annotation_list):
                if os.path.splitext(os.path.basename(annotation_list[count]))[0] == os.path.splitext(os.path.basename(annotation_list[count]))[0]:
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
            
    except BentoMLException as e:
        print(f"Error message: {e}")
        status_data_source = 'Error'
        db_mysql_stat_update(sql_select='Stat', projectId=projectId, versionId=versionId, statusOfDataSource=status_data_source)

input_spec_preprocessing = JSON(pydantic_model=PreprocessingParams)

@svc.api(input=input_spec_preprocessing, output=JSON(), route='/preprocessing')
def preprocessing(arg: PreprocessingParams):
    print('preprocessing')
    print(arg)
    print(type(arg))

    base_path = '/mnt/dlabflow/backend/minio/aiproject'
    os.makedirs(base_path, exist_ok=True)    

    arg_dataPath = [", ".join(arg.dataPath)]
    arg_dataNormalization = [", ".join(arg.dataNormalization)]
    arg_dataAugmentation = [", ".join(arg.dataAugmentation)]
    
    def db_mysql_dataframe(sql_select):
        db = pymysql.connect(host='10.40.217.236', user='root', password='password', port=3306, db='yolo', charset='utf8')
        cursor = db.cursor()
        sql = f"select * from {sql_select}"
        cursor.execute(sql)
        db.commit()
        df = pd.read_sql(sql, db)
        return df

    def db_mysql_stat_update(sql_select, projectId, versionId, statusOfProject):
        db = pymysql.connect(host='10.40.217.236', user='root', password='password', port=3306, db='yolo', charset='utf8')
        cursor = db.cursor()
        sql = f"Update {sql_select} set statusOfProject=%s where (projectId, versionId)=%s"
        val = [statusOfProject, (projectId, versionId)]
        cursor.execute(sql, val)
        db.commit()
        cursor.close()

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

    folder_name = arg.projectId+'/'+arg.versionId+'/'

    if not client.bucket_exists(bucket_name):
        client.make_bucket(bucket_name)

    empty_file_path = "/dev/null"
    empty_object_name = f"{folder_name}.keep"

    df_mysql = db_mysql_dataframe(sql_select='Stat')
    df_mysql_select = df_mysql[(df_mysql['projectId'] == arg.projectId) & (df_mysql['versionId'] == arg.versionId)]

    try:
        if 'Ready' in df_mysql_select['statusOfProject'].values:
            status_project = 'Finish'
            db_mysql_stat_update(sql_select='Stat', projectId=arg.projectId, versionId=arg.versionId, statusOfProject=status_project)
#            client.fput_object(bucket_name, empty_object_name, empty_file_path)

#            response = client.get_object(bucket_name, object_name)
#            csv_data = response.read().decode('utf-8')
#            response.close()
#            response.release_conn()
#            minio_df = pd.read_csv(io.StringIO(csv_data))
            
#            normalization_list = ast.literal_eval(minio_df['dataNormalization'].values[0])
#            augmentation_list = ast.literal_eval(minio_df['dataAugmentation'].values[0])

#            combined_list = normalization_list + augmentation_list
#            for item in combined_list:
#                split_items = item.split(', ')
#                for sub_item in split_items:
#                    folder_name = projectId+'/'+versionId+'/preprocessing/'+sub_item+'/'
#                    print(folder_name)
#                    client.fput_object(bucket_name, empty_object_name, empty_file_path)

        else:
            print('VOC annotation, image preprocessing Not Ready')

    except BentoMLException as e:
        print(f"Error message: {e}")
        status_project = 'Error'
        db_mysql_stat_update(sql_select='Stat', projectId=arg.projectId, versionId=arg.versionId, statusOfProject=status_project)
    
#    finally:
#        try:
#            if 'Finish' in df_mysql_select['statusOfDataSource'].values:
#                print('Data normalization & augmentation preprocessing Running')
#                os.system('python3 /mnt/dlabflow/backend/kubeflow/pipelines/preprocessing.py --projectId=%s --versionId=%s --dataPath=%s --dataNormalization=%s --dataAugmentation=%s --trainRatio=%s --validationRatio=%s --testRatio=%s' %(arg.projectId, arg.versionId, arg_dataPath, arg_dataNormalization, arg_dataAugmentation, arg.trainRatio, arg.validationRatio, arg.testRatio))
#            else:
#                print('VOC annotation, image preprocessing Not Finish')
#        except BentoMLException as e:
#            print(f"Error message: {e}")

input_spec_training = JSON(pydantic_model=TrainingParams)

@svc.api(input=input_spec_training, output = JSON(), route = '/training')
def training(arg: TrainingParams):
    print('training')
    print(arg)
    print(type(arg))

    os.system('python3 /mnt/dlabflow/backend/kubeflow/pipelines/training.py --projectId=%s --versionId=%s --algorithm=%s --batchsize=%s --epoch=%s' %(arg.projectId, arg.versionId, arg.algorithm, arg.batchsize, arg.epoch))

input_spec_inference = JSON(pydantic_model=InferenceParams)

@svc.api(input=input_spec_inference, output=JSON(), route='/inference')
def inference(arg: InferenceParams):
    print('inference')
    print(arg)
    print(type(arg))

    def db_mysql_dataframe(sql_select):
        db = pymysql.connect(host='10.40.217.236', user='root', password='password', port=3306, db='yolo', charset='utf8')
        cursor = db.cursor()
        sql = f"select * from {sql_select}"
        cursor.execute(sql)
        db.commit()
        df = pd.read_sql(sql, db)
        return df    

    def db_mysql_stat_update(sql_select, projectId, versionId, statusOfInference):
        db = pymysql.connect(host='10.40.217.236', user='root', password='password', port=3306, db='yolo', charset='utf8')
        cursor = db.cursor()
        sql = f"Update {sql_select} set statusOfInference=%s where (projectId, versionId)=%s"
        val = [statusOfInference, (projectId, versionId)]
        cursor.execute(sql, val)
        db.commit()
        cursor.close()

    bucket_name = 'aiproject'
    folder_name = arg.projectId+'/'+arg.versionId+'/inference/before/'

    empty_file_path = "/dev/null"
    empty_object_name = f"{folder_name}.keep"

    client = Minio(endpoint='10.40.217.236:9002', access_key='dlab-backend', secret_key='dlab-backend-secret', secure=False)
    client.fput_object(bucket_name, empty_object_name, empty_file_path)

    status_inference = 'Create'
    db_mysql_stat_update(sql_select='Stat', projectId=arg.projectId, versionId=arg.versionId, statusOfInference=status_inference)

    df_mysql = db_mysql_dataframe(sql_select='Stat')
    df_mysql_select = df_mysql[(df_mysql['projectId'] == arg.projectId) & (df_mysql['versionId'] == arg.versionId)]

    os.system('python3 /mnt/dlabflow/backend/kubeflow/pipelines/inference.py --projectId=%s --versionId=%s --inferenceAlgorithm=%s' %(arg.projectId, arg.versionId, arg.inferenceAlgorithm))

#    if 'Upload' in df_mysql_select['statusOfInference'].values: 
#        os.system('python3 /mnt/dlabflow/backend/kubeflow/pipelines/inference.py --projectId=%s --versionId=%s --inferenceAlgorithm=%s' %(arg.projectId, arg.versionId, arg.inferenceAlgorithm))
#    else:
#        print('Inference image Not Upload')
