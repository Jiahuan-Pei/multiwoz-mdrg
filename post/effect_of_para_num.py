#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Function : Compare the performance with similar size of model parameters
@License : Copyright(C), ILPS group, Univeristy of Amsterdam
@Author  : Jiahuan Pei
@Contact : j.pei@uva.nl
@Data: 2019-08-06
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

bsl = [
    (  574403, 89.5001), # bsl-150; 34970
    ( 1569653, 94.1281), # bsl-300; 37847
    ( 2638553, 89.8936), # bsl-400; 38040;
    ( 3987453, 89.3915), # bsl-500; 37844
    ( 5616353, 90.4295), # bsl-600; 38041;
    ( 7525253, 84.1860), # bsl-700; 38043;
    ( 9714153, 95.1428), # bsl-800; 38044;
    (12183053, 91.1405), # bsl-900; 38045;
    # (14931953, 91.1405), # bsl-1000; 37846; ep=19.
]

pmoe = [
    (  585699, 99.4251),      # rpmoe-30 ; 37852; ep17
    ( 1442919, 99.4251),      # rpmoe-50 ; 37850; ep19
    ( 5395469, 99.4251), # rpmoe-100; 37849; ep18
    (11933019, 99.4251), # rpmoe-150; 36281
]

bsl_df = pd.DataFrame(bsl, columns=["num", "score"]).set_index('num', drop=False)
bsl_df['score'].plot()
pmoe_df = pd.DataFrame(pmoe, columns=["num", "score"]).set_index('num', drop=False)
pmoe_df['score'].plot()
print(bsl_df.head())
print(pmoe_df.head())
plt.xlabel('Number of parameters')
plt.ylabel('Score')
plt.legend(['BSL', 'RPMOE'], loc=1)
plt.tight_layout()
plt.show()

if __name__ == "__main__":
    pass