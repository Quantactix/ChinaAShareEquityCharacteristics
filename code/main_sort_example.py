
import pickle as pkl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def read_pickle_file(file):
    pickle_data = pd.read_pickle(file)
    return pickle_data

cs_vw = read_pickle_file('./data/share/cs_vw.pkl')
cs_ew = read_pickle_file('./data/share/cs_ew.pkl')

# # Print Returns

uni_ret = pd.DataFrame()
for c in cs_vw.char_list:
    expr = "tmp=cs_vw.port_%s.uni_ret.copy()"%(c)
    exec(expr)
    tmp.columns = [c+i for i in tmp.columns]
    tmp.index=pd.to_datetime(tmp.index)
    uni_ret=uni_ret.append(tmp.T)
    del tmp
    
uni_ret.T.to_csv('./data/share/sorted_portfolio_vw/unisort_returns.csv')

uni_sr = [] 
uni_ls_ret = pd.DataFrame()
for c in cs_vw.char_list:
    expr = "tmp=cs_vw.port_%s.uni_ls.copy()"%(c)
    exec(expr)
    tmp.index=pd.to_datetime(tmp.index)
    expr = "uni_ls_ret['%s']=tmp"%(c)
    exec(expr)
    expr = "uni_sr.append(cs_vw.port_%s.uni_ls_sr)"%(c)
    exec(expr)
    del tmp

uni_sr = pd.Series(uni_sr, index=cs_vw.char_list)
uni_ls_ret.to_csv('./data/share/sorted_portfolio_vw/unifactor_returns.csv')

bi_sr = [] 
bi_ret = pd.DataFrame()
for c in cs_vw.char_list[1:]:
    expr = "tmp=cs_vw.port_%s.bi_ret.copy()"%(c)
    exec(expr)
    tmp.index=pd.to_datetime(tmp.index)
    bi_ret=bi_ret.append(tmp.T)
    expr = "bi_sr.append(cs_vw.port_%s.bi_ls_sr)"%(c)
    exec(expr)
    del tmp
    
bi_ret.T.to_csv('./data/share/sorted_portfolio_vw/bisort_returns.csv')


# # EW

uni_ret = pd.DataFrame()
for c in cs_ew.char_list:
    # expr = "cs_ew.port_%s.uni_ret.to_csv('./data_Jan2021/chars60/sorted_portfolio_ew/unisort/unisort_ew_%s.csv')"%(c,c)
    expr = "tmp=cs_ew.port_%s.uni_ret.copy()"%(c)
    exec(expr)
    tmp.columns = [c+i for i in tmp.columns]
    tmp.index=pd.to_datetime(tmp.index)
    uni_ret=uni_ret.append(tmp.T)
    del tmp
    
uni_ret.T.to_csv('./data/share/sorted_portfolio_ew/unisort_returns.csv')

uni_sr = [] 
uni_ls_ret = pd.DataFrame()
for c in cs_ew.char_list:
    # expr = "cs_ew.port_%s.uni_ret.to_csv('./data_Jan2021/chars60/sorted_portfolio_ew/unisort/unisort_ew_%s.csv')"%(c,c)
    expr = "tmp=cs_ew.port_%s.uni_ls.copy()"%(c)
    exec(expr)
    tmp.index=pd.to_datetime(tmp.index)
    expr = "uni_ls_ret['%s']=tmp"%(c)
    exec(expr)
    expr = "uni_sr.append(cs_ew.port_%s.uni_ls_sr)"%(c)
    exec(expr)
    del tmp

uni_sr = pd.Series(uni_sr, index=cs_ew.char_list)
uni_ls_ret.to_csv('./data/share/sorted_portfolio_ew/unifactor_returns.csv')

bi_sr = [] 
bi_ret = pd.DataFrame()
for c in cs_ew.char_list[1:]:
    expr = "tmp=cs_ew.port_%s.bi_ret.copy()"%(c)
    exec(expr)
    tmp.index=pd.to_datetime(tmp.index)
    bi_ret=bi_ret.append(tmp.T)
    expr = "bi_sr.append(cs_ew.port_%s.bi_ls_sr)"%(c)
    exec(expr)
    del tmp
    
bi_ret.T.to_csv('./data/share/sorted_portfolio_ew/bisort_returns.csv')
