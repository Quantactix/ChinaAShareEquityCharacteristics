import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pandas.tseries.offsets import MonthBegin, MonthEnd
from tqdm import tqdm

# import warnings
# warnings.filterwarnings("ignore")

# list of variables

char_list = ['abr','adm','beta','bm','cfp',
             'dbeta','de','ep','idvc','m1',
             'm11','rdm','roe','rs','season',
             'sg','size','sp','sue','tv'
            ]

id_list = ['asset', 'date', 'ret', 'xret', 'lag_me', 'log_me']

ts_list = ['rf_mon', 'mktrf', 'VMG', 'SMB', 'PMO']

# read data

df = pd.read_csv("./data/csmar_tables/TRD_Mnth.csv")
#,encoding='utf8',error_bad_lines=False, engine ='python')
df = df[['Stkcd','Trdmnt','Mretwd']]
df.columns = ['asset','date','ret']
df['date'] = pd.to_datetime(df['date'])  # date formated
df['date'] = df['date']+MonthEnd(0)      # date is formatted as month end
df = df[ (df['date']>='19910101') & (df['date']<='20201231')]

df_pivot = pd.pivot(data=df,values='ret', index='date', columns='asset')
df_pivot = df_pivot.reset_index()

# # cross-sectional 
# 
# - asset characteristics

for char in tqdm(char_list):
    da = pd.read_csv("./data/chars/"+char+".csv")
    #,encoding='utf8',error_bad_lines=False, engine ='python')
    da['Trdmnt'] = pd.to_datetime(da['Trdmnt'],format='%Y%m')
    da['Trdmnt'] = da['Trdmnt'] + MonthEnd(0) # the end of this month
    da['Trdmnt'] = da['Trdmnt'] + MonthEnd(1) # the end of next month
    # da = da[ (da['Trdmnt']>='2000') & (da['Trdmnt']<='2020')] # no need to cut the data, merging with drop the date out of range.
    df_melt = da.melt(id_vars=['Trdmnt'],value_name =char)
    df_melt.columns = ['date','asset',char]
    outputpath="./data/tmp/"+char+".csv"
    df_melt.to_csv(outputpath,sep=',',index=False,header=True)

da = df.copy()
for char in tqdm(char_list):
    s1 = pd.read_csv("./data/tmp/"+char+".csv")
    #,encoding='utf8',error_bad_lines=False, engine ='python')
    s1['date'] = pd.to_datetime(s1['date'])
    da = pd.merge(da,s1,how='left',on=['date','asset'])

# # time-series variables
# 
# - factors
# - macro predictors

f= pd.read_csv("./data/factors/ch4/CH_4_fac_update_20211231.csv",skiprows=9)
#,encoding='utf8',error_bad_lines=False, engine ='python')
f.rename(columns={'mnthdt':'date'}, inplace = True)
f['date'] = pd.to_datetime([str(i) for i in f['date']])
for i in f.columns[1:]:
    print(i)
    f[i]=f[i]/100

da = pd.merge(da,f,how='left',on=['date'])
da['xret']=da['ret']-da['rf_mon']

da['lag_me'] = np.exp(da['size'])
da['log_me'] = da['size']

# # output the raw data

da=da[id_list + char_list + ts_list]

da=da[~da['xret'].isna()]
da=da[~da['size'].isna()]

da.to_csv("./data/share/panel_raw.csv")

# standardize the cross-sectional characteristics
# rank the characteristics

def standardize(df):
    # exclude the the information columns
    col_names = df.columns.values.tolist()
    list_to_remove = ['asset', 'date',
                      'ret', 'xret', 'lag_me', 'log_me', 'rf_mon', 'mktrf', 'VMG', 'SMB', 'PMO'
                     ]
    
    col_names = list(set(col_names).difference(set(list_to_remove)))
    print(col_names)
    for col_name in tqdm(col_names):
        # print('processing %s' % col_name)
        # count the non-missing number of factors, we only count non-missing values
        unique_count = df.dropna(subset=['%s' % col_name]).groupby(['date'])['%s' % col_name].unique().apply(len)
        unique_count = pd.DataFrame(unique_count).reset_index()
        unique_count.columns = ['date', 'count']
        df = pd.merge(df, unique_count, how='left', on=['date'])
        # ranking, and then standardize the data
        df['%s_rank' % col_name] = df.groupby(['date'])['%s' % col_name].rank(method='dense')
        df['rank_%s' % col_name] = (df['%s_rank' % col_name] - 1) / (df['count'] - 1) * 2 - 1
        df = df.drop(['%s_rank' % col_name, '%s' % col_name, 'count'], axis=1)
    df = df.fillna(0)
    return df

da_rank = standardize(da)

# # output the rank data
da_rank.to_csv("./data/share/panel_rank.csv")
