#! /bin/bash

wget -q https://storage.yandexcloud.net/datasouls-ods/materials/f90231b6/items.csv -O "$1/items.csv"
wget -q https://storage.yandexcloud.net/datasouls-ods/materials/6503d6ab/users.csv -O "$1/users.csv"
wget -q https://storage.yandexcloud.net/datasouls-ods/materials/04adaecc/interactions.csv -O "$1/interactions.csv"
wget -q https://storage.yandexcloud.net/datasouls-ods/materials/faa61a41/sample_submission.csv -O "$1/sample_submission.csv"

