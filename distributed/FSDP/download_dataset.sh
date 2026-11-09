#!/bin/bash

# Create the "data" folder if it doesn't exist
# 若 data 目录不存在，则创建该目录。
mkdir -p data

# Download the files into the "data" folder
# 将文件下载到 data 目录。
wget -P data https://public-nlp-datasets.s3.us-west-2.amazonaws.com/wikihowAll.csv
wget -P data https://public-nlp-datasets.s3.us-west-2.amazonaws.com/wikihowSep.csv
