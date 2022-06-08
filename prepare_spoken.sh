#!/bin/bash

python spoken_preprocess.py --data-dir /data3/sgahn/NIKL_DAILY_CONVERSATION_2020_v1.2_text/ --save-dir /data3/sgahn/NIKL_DAILY_CONVERSATION_2020_v1.2_text_kss/ --do-rearrange  # done
# python spoken_preprocess.py --data-dir /data3/sgahn/NIKL_KParlty_2021_v1.0/ --save-dir /data3/sgahn/NIKL_KParlty_2021_v1.0_txt/ --do-rearrange
python spoken_preprocess.py --data-dir /data3/sgahn/NIKL_MESSENGER_v2.0_text/ --save-dir /data3/sgahn/NIKL_MESSENGER_v2.0_text_kss/ --do-rearrange  # done
python spoken_preprocess.py --data-dir /data3/sgahn/NIKL_OM_2021_v1.0_text/ --save-dir /data3/sgahn/NIKL_OM_2021_v1.0_text_kss/ --do-rearrange
python spoken_preprocess.py --data-dir /data3/sgahn/NIKL_SPOKEN_text/ --save-dir /data3/sgahn/NIKL_SPOKEN_text_kss/ --do-rearrange
