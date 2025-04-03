#!/usr/bin/env python
# coding: utf-8

# ### S3, Redshift : 데이터 import 

import boto3
import redshift_connector
import pandas as pd
import time
import pickle
import gzip
s3 = boto3.client('s3')
response = s3.list_buckets()
buckets = [bucket['Name'] for bucket in response['Buckets']]



start = time.time()

### redshift 연결
conn = redshift_connector.connect(
    host='호스트 입력',
    database='DB 입력',
    user='USERNAME입력',
    password='PASSWORD입력',
    port = '포트번호입력'
)

### Redshift에서 가져올 Select 쿼리 정의    --> 기존에 들어와있고 새롭게 이미지 생성된 애들만 가져오기 위함임
#-- """ """ 사이에 쿼리 입력 필요 - Tab이나 줄변경 시 error를 피하기 위함
query = """
  쿼리는 알아서 입력하긩~
;
"""

### 지정한 쿼리 실행 후 df 객체에 입력
cursor: redshift_connector.Cursor = conn.cursor()
cursor.execute(query)

data: tuple = cursor.fetchall()
df = pd.DataFrame.from_records(data)
df.columns = [x[0].decode() for x in cursor.description]


### End Time 체크 - 데이터 불러오는데 시간 체크(End Time - Start Time)
print(time.time()-start)

### Redshift DB에서 가져온 데이터 확인
df.head(3)

df['prd_img_name_url'] = df.url + df.prd_img_nm 


# ### image color clustering 


import pandas as pd 
import numpy as np 
from color_extractor import color_extractor  
df.head(3)
func = color_extractor()


# df_400000 변경 
bucket='버킷명'
size = (100,100)    ## 이건 변경해도 괜찮을 듯 
result, percent, rgb, check_img = func.image_result(bucket, df.prd_img_name_url, size)
## 생성값 df에 추가 
df['rgb_with_percent'] = result
df['percent'] = percent
df['rgb'] = rgb
df['error'] = check_img


rgb_final, hex_final = func.hex_rgb_cng(df)
df.rgb = rgb_final 
df['hex'] = hex_final 

##s3 경로에 넣어주자 
df.to_csv("S3경로")


print("WORK FINISH python code Exit")

exit()
