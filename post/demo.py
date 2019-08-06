#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Function : A script for removing useless logs
@License : Copyright(C), ILPS group, Univeristy of Amsterdam
@Author  : Jiahuan Pei
@Contact : j.pei@uva.nl
@Data: 2019-07-21
"""

import os, sys, shutil
clean_dir = sys.argv[1]
err_path = '%s/%s'%(clean_dir, 'job_errors')
out_path = '%s/%s'%(clean_dir, 'job_out')
res_path = '%s/%s'%(clean_dir, 'results')
for fe in os.listdir(err_path):
    if os.path.getsize('%s/%s'%(err_path, fe))>0: # has error
        fo = fe.replace('.error', '.out')
        rd = fe.replace('.error', '')
        os.remove('%s/%s'%(err_path, fe))
        shutil.rmtree('%s/%s'%(res_path, rd))
        print('deleted:', fe)
        try:
            os.remove('%s/%s'%(out_path, fo))
            print('deleted:', fo)
        except:
            pass


if __name__ == "__main__":
    pass