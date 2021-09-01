FROM python:3.6-buster

ENV AWS_DEFAULT_REGION=ap-northeast-2

#pip 업그레이드
RUN python -m pip install pip --upgrade

# 현재 폴더내의 모들 파일들을 이미지에 추가
ADD . /app

# 작업 디렉토리로 이동
WORKDIR /app

# 작업 디렉토리에 있는 requirements.txt로 패키지 설치
RUN pip install -r requirements.txt

# 컨테이너에서 실행될 명령어. 컨테이거나 실행되면 train하고 deploy.
CMD python main.py --TransE
