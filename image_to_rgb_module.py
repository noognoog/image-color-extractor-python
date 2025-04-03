import pandas as pd 
import numpy as np 
from sklearn.cluster import KMeans 
from sklearn.cluster import MiniBatchKMeans
import os 
import random
import cv2
from tqdm import tqdm_notebook as tqdm
from PIL import Image 
from PIL import ImageFile 
import seaborn as sns 
import matplotlib.pyplot as plt 
import boto3
import warnings 
warnings.filterwarnings(action='ignore')

class image_to_rgb : 
    ###aws s3 버킷에서 이미지를 읽어오는 함수 정의 
    def read_image_from_s3(self, bucket, key):
    
        ImageFile.LOAD_TRUNCATED_IMAGES = True         ##이미지 파일을 로드할 때 이미지 파일의 끝이 손실된 경우에도 로드할 수 있게 하기 위해 추가 
        s3 = boto3.resource('s3')                      ##s3 리소스 객체 생성 
        bucket = s3.Bucket(bucket)                     ##s3 버킷 객체 생성 
        object = bucket.Object(key)
        try : 
            response = object.get()
            file_stream = response['Body']
            im = Image.open(file_stream)               ##PIL라이브러리를 사용하여 파일 스트림에서 이미지를 열기
        except:
            im = [] 
       # im = im.resize(size)
        return np.array(im)                            ##최종적으로 로드된 이미지를 numpy 배열로 변환하고 반환함. 이미지 로드 실패하면 빈 리스트 반환
  
    ##이미지에 마스킹을 적용하는 함수 정의.
    def mask_image(self, img) : 
        
        # img : 이미지 
        
        height, width = img.shape[:2]                  ##이미지의 높이와 너비를 받음
        mask = np.zeros(img.shape[:2],np.uint8)        ##입력 이미지와 같은 크기의 빈 마스크 이미지 생성 (어떤 부분 투명하게 할 지 정하기 위해 )
        bgdModel = np.zeros((1,65),np.float64)         ##그랩컷 알고리즘이 사용할 배경 및 전경 모델 초기화. 
        fgdModel = np.zeros((1,65),np.float64)
        rect = (10,10,width-30,height-30)              ##이미지 내에서 마스킹을 적용할 사각형 영역 정의
        cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)  ##그랩컷 알고리즘을 사용해서 이미지에 마스킹 적용. 배경과 전경 구분하고 마스크 업데이트 
        mask = np.where((mask==2)|(mask==0),0,1).astype("uint8")              ##그랩컷 알고리즘에 의해 생성된 마스크를 수정해서 배경과 무관한 부분은 0으로, 전경은 1로 설정
        img1 = img*mask[:,:,np.newaxis]                                       ##원본 이미지에 마스킹 적용해서 마스킹된 이미지 생성. 
        return img1    
    
    ##던 지수를 계산하는 함수 정의.(for 군집화 결과 품질 평가)
    def dunn_index(self, centroids):
        
        # centroids : clt.cluster_centers_
        
        from scipy.spatial import distance_matrix
        cluster_dmat = distance_matrix(centroids, centroids)
        # fill diagonal with +inf: ignore zero distance to self in "min" computation
        np.fill_diagonal(cluster_dmat, np.inf)
        min_intercluster_dist = cluster_dmat.min()
        return min_intercluster_dist     

    ##이미지의 색상을 기반으로 클러스터링을 수행하는 함수 정의. 이 함수 호출하면 이미지의 색깔을 기준으로 클러스터링된 모델 반환
    def clustering_color_base(self, img): 

        change1 = img.reshape((img.shape[0] * img.shape[1], 3))                   ##입력 이미지를 2D배열에서 1D배열로 변환. 

        i = 1 
        dunn = 100 

        while i < 8 and dunn > 60 :                                               ##클러스터 수 i가 8보다 작고 dunn지수가 60보다 큰 경우에만 반복
            clt = MiniBatchKMeans(n_clusters = i, random_state=0, max_iter=10)    ##클러스터링 수행 
            clt.fit(change1)
            dunn = self.dunn_index(clt.cluster_centers_)                          ##현재 클러스터 수로 계산된 던 지수 산출
            result = i-1 
            i += 1 
        clt = MiniBatchKMeans(n_clusters = result, random_state=0, max_iter=10)   ##현재 클러스터 수를 기록하고 다음 클러스터 수를 계산하기 위해 i 증가
        clt.fit(change1)

        return clt     
    ##두 개의 이미지를 비교해서 배경을 제거한 이미지(img1)와 원본이미지(img)에서 동일한 색상 픽셀 추출 후, 추출된 픽셀을 기반으로 클러스터링 수행하는 함수 정의. 
    ##이미지 내에서 유사한 색상을 클러스터링해서 객체를 식별하는데 사용 
    def clustering_color(self, img, img1): 
        
        #img : 이미지 
        #img1 : 배경제거 이미지 == mask_image(img)

        change1 = img.reshape((img.shape[0] * img.shape[1], 3))            ##1D배열로 변환
        change2 = img1.reshape((img1.shape[0] * img1.shape[1], 3))         ##1D배열로 변환 
        change = change1[(change1[:,0] == change2[:,0]) & (change1[:,1] == change2[:,1]) & (change1[:,2] == change2[:,2])]   ##change1과 change2에서 동일한 RGB색상을 가진 픽셀만 추출해서 change에 저장
        ##위의 과정을 통해 배경 제거된 이미지랑 원본 이미지에서 동일한 색상을 가진 픽셀만 사용해서 클러스터링 수행 
        i = 1         ##클러스터링 초기값 설정 
        dunn = 100    ##dunn지수 초기값 설정 
        
        while i < 8 and dunn > 60 :       ##클러스터 수가 8미만이고 dunn 지수가 60이상일 때까지 반복
            clt = MiniBatchKMeans(n_clusters = i, random_state=0, max_iter=10) #, n_jobs = -1   ##MinibatchKmeans를 사용하여 클러스터링 수행. 클러스터 수가 result에 저장됨
            clt.fit(change)
            dunn = self.dunn_index(clt.cluster_centers_) 
            result = i-1 
            i += 1 
        clt = MiniBatchKMeans(n_clusters = result, random_state=0, max_iter=10)
        clt.fit(change)

        return clt  
    ##kmeans 클러스터링 모델에서 얻은 클러스터링 결과를 기반으로 클러스터의 중심에 대한 히스토그램 생성하는 함수 정의. 
    def centroid_histogram(self, clt):
        numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
        (hist, _) = np.histogram(clt.labels_, bins=numLabels)

        hist = hist.astype("float")
        hist /= hist.sum()

        return hist 
    
    ##이미지에서 추출한 색상 히스토그램과 해당 색상 클러스터의 중심점을 사용해서 색상 막대 그래프를 그리는 기능을 제공
    def plot_colors(self, hist, centroids):
        bar = np.zeros((50, 300, 3), dtype="uint8")
        startX = 0

        for (percent, color) in zip(hist, centroids):              
            endX = startX + (percent * 300)                        
            cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),  
                          color.astype("uint8").tolist(), -1)
            startX = endX
            
        return bar