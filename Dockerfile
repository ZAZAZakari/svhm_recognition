FROM ubuntu:14.04

RUN apt-get -y update

RUN apt-get -y install python-software-properties python g++ make
RUN apt-get -y install build-essential gfortran libatlas-base-dev python-pip python-dev