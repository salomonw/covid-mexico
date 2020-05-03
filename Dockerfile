# Dockerfile
#
# Parent iamge
FROM ubuntu:16.04

# Required installations
RUN apt-get -y update && apt-get install -y \
  python3 \
  python3-pip \
  make \
  unzip \
  mercurial \
  wget \
  libgmp3-dev \
  vim

RUN pip3 install --upgrade pip
RUN apt-get -y update
RUN apt-get -y upgrade
RUN pip3 install numpy
RUN pip3 install scipy
RUN pip3 install matplotlib
RUN pip3 install seaborn
RUN pip3 install sklearn
RUN pip3 install xgboost
RUN pip3 install tensorflow
RUN pip3 install datetime


WORKDIR /
#RUN mkdir app



# CMD ["./start_script.sh"]