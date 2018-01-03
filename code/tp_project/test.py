# -*- coding: utf-8 -*-
'''
Created on 2017年12月20日

@author: Qiaowei
'''

import handle_features as hf

date = '2018-01-02'
system = 'BOCOP-*'
res = hf.get_data(date, system, hourfrom='00:00:00', hourto='00:30:00')

print(res)
print(len(res))