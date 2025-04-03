# 이미지 색상 추출기(python with AWS S3, AWS Redshift)

## 프로젝트 소개
AWS S3에 저장되어있는 이미지를 가져와 클러스터링을 통해 주요 색상 정보를 추가하는 모듈입니다.
<br/>
<br/>

## 주요 기능
*   **이미지 색상 추출**: AWS S3 버킷에 저장된 이미지에서 주요 색상을 추출
*   **색상 클러스터링**: 추출된 색상 데이터를 클러스터링하여 이미지의 주요 색상 그룹을 식별
*   **RGB 및 HEX 색상 코드 변환**: 추출된 색상 데이터를 RGB 및 HEX 코드로 변환
*   **데이터베이스 연동**: AWS Redshift 데이터베이스와 이미지 메타데이터 연동
*   **결과 저장**: 추출된 색상 정보와 분석 결과를 CSV 파일로 저장
<br/>
<br/>

## 기술 스택
*   Python
*   boto3
*   redshift\_connector
*   pandas
*   scikit-learn
*   opencv-python
*   Pillow
<br/>
<br/>

## 구조
1.  **`image_to_rgb_module.py`**: 기본적인 이미지 처리 및 색상 추출 기능을 가지고 있는 클래스 
2.  **`color_extractor_module.py`**: `image_to_rgb_module.py`을 확장하여, 이미지에서 색상 정보를 추출하고 클러스터링하는 기능을 제공하는 클래스
3.  **`image_color_pipeline.py`**: Redshift에서 이미지 메타데이터를 가져오고, S3에서 이미지를 다운로드하여 `color_extractor_module.py`의 기능을 통해 색상 데이터를 처리하고, 결과 데이터를 최종적으로 CSV 파일로 저장
<br/>
<br/>

## 실행 방법
1.  필요한 라이브러리를 설치
    ```
    pip install boto3 redshift_connector pandas numpy scikit-learn opencv-python pillow
    ```
2.  AWS S3와 Redshift 연결 설정을 구성
3.  `image_color_pipeline.py`를 실행합니다:

    ```
    python image_color_pipeline.py
    ```
<br/>
<br/>
## 결과물
프로그램 실행 결과로 이미지의 주요 색상 정보(RGB, HEX 코드)가 담긴 CSV 파일이 생성됨 
