#! /bin/bash

DATADIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
mkdir -p "$DATADIR/data"

wget -q https://storage.yandexcloud.net/datasouls-ods/materials/f90231b6/items.csv -O "$DATADIR/data/items.csv"
wget -q https://storage.yandexcloud.net/datasouls-ods/materials/6503d6ab/users.csv -O "$DATADIR/data/users.csv"
wget -q https://storage.yandexcloud.net/datasouls-ods/materials/04adaecc/interactions.csv -O "$DATADIR/data/interactions.csv"
wget -q https://storage.yandexcloud.net/datasouls-ods/materials/faa61a41/sample_submission.csv -O "$DATADIR/data/sample_submission.csv"

