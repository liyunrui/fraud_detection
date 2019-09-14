FROM python:3.7.4

ENV APPDIR=/usr/local/app

RUN mkdir -p APPDIR
WORKDIR ${APPDIR}

ADD data/train.zip .
ADD data/test.zip .

RUN unzip train.zip
RUN unzip test.zip

ADD requirements.txt .
RUN pip install -r requirements.txt

ADD src ./src
RUN mkdir result
