#!/bin/bash

python prepare_dataset.py --data-dir /data/sgahn/NIKL_NEWSPAPER/ --save-dir /data/sgahn/NIKL_NEWSPAPER_txt/
python prepare_dataset.py --data-dir /data/sgahn/NIKL_NEWSPAPER_2020/ --save-dir /data/sgahn/NIKL_NEWSPAPER_2020_txt/
python prepare_dataset.py --data-dir /data/sgahn/NIKL_NEWSPAPER_2021_v1.0/ --save-dir /data/sgahn/NIKL_NEWSPAPER_2021_v1.0_txt/
python prepare_dataset.py --data-dir /data/sgahn/NIKL_WRITTEN/ --save-dir /data/sgahn/NIKL_WRITTEN_txt
