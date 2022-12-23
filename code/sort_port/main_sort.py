#!/usr/bin/env python
# coding: utf-8

# - 61 unisort
# - 60 bisort
# 
# all covered in the code: tools

# In[1]:


from tools.crosssection import *


# ### read data

# In[2]:


file = '../../share/panel_rank.csv'
df = pd.read_csv(file)

df.columns

# In[4]:


char_list = [
    'adm', 'beta', 'bm', 'ch4mkt_beta',
    'ch4pmo_beta', 'ch4smb_beta', 'ch4vmg_beta', 'dbeta', 'de', 'ep',
    'idvc', 'm1', 'm11', 'rdm', 'rs', 'season', 'me', 'sp', 'sue', 'tv'
]
# char_list = ['roe','mom12m','beta']

# df['lag_me'] = df['size']
# df['log_me'] = np.log(df['size'])

print(len(char_list))
rank_char_list = ['rank_'+i for i in char_list]

# identifiers
ids = ['asset','date','ret','xret','lag_me','log_me']


# ### delete obs. with NA me

# In[5]:


df1 = df[ids+rank_char_list]
print(df1.shape)
df1 = df1[~df1['lag_me'].isna()]
print(df1.shape)


# In[6]:


print(df1.head())


# # 20210128
# - do sorting based on the rank
# - construct decile portfolios
# - calculate the ls factors

# In[ ]:

# vw

cs_vw = cs(df1, char_list, 10, 'lag_me')
cs_vw.update_all(parallel=True)

with open('../../share/cs_vw.pkl', 'wb') as f:
    pkl.dump(cs_vw, f)

del cs_vw

# ew

cs_ew = cs(df1, char_list, 10, 'ew')
cs_ew.update_all(parallel=True)

with open('../../share/cs_ew.pkl', 'wb') as f:
    pkl.dump(cs_ew, f)

del cs_ew