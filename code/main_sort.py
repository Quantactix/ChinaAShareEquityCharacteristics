
from tools.crosssection import *

file = './data/share/panel_rank.csv'
df = pd.read_csv(file)

char_list = ['size',
            'abr','adm','beta','bm','cfp',
             'dbeta','de','ep','idvc','m1',
             'm11','rdm','roe','rs','season',
             'sg','sp','sue','tv'
            ]

print(len(char_list))
rank_char_list = ['rank_'+i for i in char_list]

# identifiers
ids = ['asset','date','ret','xret','lag_me','log_me']


# ### delete obs. with NA me

df1 = df[ids+rank_char_list]
print(df1.shape)
df1 = df1[~df1['lag_me'].isna()]
print(df1.shape)

# vw

cs_vw = cs(df1, char_list, 10, 'lag_me')
cs_vw.update_all(parallel=True)

with open('./data/share/cs_vw.pkl', 'wb') as f:
    pkl.dump(cs_vw, f)

del cs_vw

# ew

cs_ew = cs(df1, char_list, 10, 'ew')
cs_ew.update_all(parallel=True)

with open('./data/share/cs_ew.pkl', 'wb') as f:
    pkl.dump(cs_ew, f)

del cs_ew