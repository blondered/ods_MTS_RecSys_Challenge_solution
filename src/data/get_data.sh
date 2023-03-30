#! /bin/bash

echo "Downloading data"

mkdir -p "data/raw"
mkdir -p "data/interim"
mkdir -p "data/processed"

wget -q https://storage.yandexcloud.net/datasouls-ods/materials/f90231b6/items.csv -O "data/raw/items.csv"
wget -q https://storage.yandexcloud.net/datasouls-ods/materials/6503d6ab/users.csv -O "data/raw/users.csv"
wget -q https://storage.yandexcloud.net/datasouls-ods/materials/04adaecc/interactions.csv -O "data/raw/interactions.csv"
wget -q https://storage.yandexcloud.net/datasouls-ods/materials/faa61a41/sample_submission.csv -O "data/raw/sample_submission.csv"

