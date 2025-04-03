import pandas as pd 
import numpy as np 
from sklearn.cluster import KMeans 
import os 
import random
import cv2
from tqdm import tqdm_notebook as tqdm
from PIL import Image 
from io import BytesIO
import seaborn as sns 
import matplotlib.pyplot as plt 
from image_to_rgb import image_to_rgb
import webcolors 

class color_extractor(image_to_rgb): 
    
    ##이미지를 처리하고 시각적으로 표시하기 위한 함수 정의(결과 이미지 보기 위함)
    ##aws s3 버킷에서 이미지 데이터를 가져와서 처리하고 시각화하는데 사용
    def image_show(self, bucket, data, size, mask_img = True) : 
      
        img = self.read_image_from_s3(bucket,data) 
        
        
        if img.size > 0 :                                           ##이미지 데이터가 유효한지 확인
            if len(img.shape) == 2  or img.shape[2] != 2 :          ##2차원인지 아닌지 확인하기. 3차원이 아니라면 BGR에서 RGB로 색상 채널 변경하고 크기 조정. 이미지 데이터 형식을 일관되게 처리하기 위해
              #  img = cv2.imread(url + data)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, size, interpolation = cv2.INTER_AREA)
                img1 = self.mask_image(img)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
                ##true인 경우, 즉 마스킹 할 게 있으면 마스킹한 이미지와 원래 이미지 두개를 바탕으로 클러스터링하고 아니면 그냥 클러스터링 
                if mask_img :                                   
                    clt = self.clustering_color(img, img1)      ##img : 본래 이미지 img1 : 마스크 처리, 다른 부분이 상품  
                else : 
                    clt = self.clustering_color_base(img)     
                
                hist = self.centroid_histogram(clt)
                bar = self.plot_colors(hist, clt.cluster_centers_)
                plt.figure()
                plt.axis("off")
                plt.imshow(img)
                plt.figure()
                plt.axis("off")
                plt.imshow(bar)
                plt.show()
                print(hist)

            for i, center in enumerate(clt.cluster_centers_):
                print("Cluster" + str(i), center)

     ###여러 이미지를 처리하고 그 결과를 데이터로 반환하기 위한 함수 정의 
    def image_result(self, bucket, data_list, size) : 
        from collections import defaultdict
        from tqdm import tqdm

        mydict = defaultdict()
        percent = list() 
        rgb = list()
        result = list() 
        check_img = list() 
        ##각 이미지 파일에 대한 루프 시작
        for filename in tqdm(data_list) : 
            mydict = defaultdict()
            
            img = self.read_image_from_s3(bucket,filename) 
            if img.size > 0 : 
                if  len(img.shape) == 2  or img.shape[2] != 2   : 
                  #  img = cv2.imread(url + filename)
                    # rgb로 변경
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, size, interpolation = cv2.INTER_AREA)
                    # background 제거를 위해 mask 씌우기 
                    img1 = self.mask_image(img)
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
                
                    # background제거 후 clustering 
                    try : 
                        clt = self.clustering_color(img, img1)
                    except : 
                        clt = self.clustering_color_base(img)
                        
                    # clustering 비중 구하기 
                    hist = self.centroid_histogram(clt)
                    ##클러스터링 결과를 딕셔너리에 저장함. 각 클러스터의 히스토그램 값으로 딕셔너리 키를 만들고, 해당 클러스터의 중심 좌표를 값으로 설정
                    for idx, center in enumerate(clt.cluster_centers_)  : 
                        mydict[hist[idx]] = list(center)
                    mydict = dict(sorted(mydict.items(), key = lambda item: item[0], reverse = True))  ###히스토그램을 기반으로 딕셔너리를 히스토그램 값에 따라 내림차순으로 정렬
                    result.append(mydict)
                    percent.append(list(mydict.keys()))
                    rgb.append(list(mydict.values()))
                    check_img.append(0)                   ##유효한 이미지는 0으로 표시
                else : 
                    result.append(np.nan)
                    percent.append(np.nan)
                    rgb.append(np.nan) 
                    check_img.append(1)                   ##유효하지 않은 이미지에 대한 결과값 리스트에 저장
            else : 
                result.append(np.nan)
                percent.append(np.nan)
                rgb.append(np.nan) 
                check_img.append(1) 
        return result, percent, rgb, check_img            ##모든 이미지 파일 처리가 완료되면 결과, 히스토그램 값, RGB 값, 이미지 유효성을 담고 있는 리스트들을 반환
        


    ##데이터프레임df의 rgb_with_percent 열에 있는 RGB 값과 해당 백분율을 가져와서 RGB와 HEX로 변환하는 함수
    def hex_rgb_cng(self, df) :
        
         ##df : RGB값과 해당 백분율이 포함된 데이터 프레임
        rgb_final = list() 
        hex_final = list() 
        for x in tqdm(df.rgb_with_percent) :                           ##image_color_final.py에서 위의 image_result 함수를 호출하고 나온 결과값을 rgb_with_percent 컬럼으로 할당하는 부분이 있음
            try : 
                dict_rgb = list(x.values())
                rgb_result = str() 
                hex_result = list() 
                for i, rgb in enumerate(dict_rgb) :
                    cc = list(np.round(rgb).astype(int))
                    hex_result.append(webcolors.rgb_to_hex(cc))
                    if i == 0 : 
                        rgb_result = str(cc) 
                    else : 
                        rgb_result = rgb_result + ' o ' + str(cc) 
                rgb_final.append(rgb_result)
                hex_final.append(hex_result)
            except : 
                rgb_final.append(np.nan)
                hex_final.append(np.nan) 
        return rgb_final, hex_final 

