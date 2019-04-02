#!/usr/bin/env bash
python train.py --data_dir=../multiwoz-moe/data --intent_type=domain_act --emb_size=20 --hid_size_enc=20 --hid_size_dec=20 --hid_size_pol=20