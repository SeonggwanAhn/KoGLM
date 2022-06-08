#!/bin/bash

python prepare_dataset.py --data-dir /data/sgahn/NIKL_DAILY_CONVERSATION_2020_v1.2/ --save-dir /data/sgahn/NIKL_DAILY_CONVERSATION_2020_v1.2_text/ --do-rearrange  # done
# python prepare_dataset.py --data-dir /data/sgahn/NIKL_KParlty_2021_v1.0/ --save-dir /data/sgahn/NIKL_KParlty_2021_v1.0_text/ --do-rearrange
python prepare_dataset.py --data-dir /data/sgahn/NIKL_MESSENGER_v2.0/ --save-dir /data/sgahn/NIKL_MESSENGER_v2.0_text/ --do-rearrange  # done
python prepare_dataset.py --data-dir /data/sgahn/NIKL_OM_2021_v1.0/ --save-dir /data/sgahn/NIKL_OM_2021_v1.0_text/ --do-rearrange
python prepare_dataset.py --data-dir /data/sgahn/NIKL_SPOKEN/ --save-dir /data/sgahn/NIKL_SPOKEN_text/ --do-rearrange
