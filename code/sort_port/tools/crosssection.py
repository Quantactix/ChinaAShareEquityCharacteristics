import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tools.others import *
from tools.sortedportfolio import port

from tqdm import tqdm
import multiprocessing as mp
import time
import pickle as pkl

import pyarrow.feather as feather

def read_feather_file(file):
    with open(file, 'rb') as f:
        data = feather.read_feather(f)
    return data

class cs:
	'''

	'''
	def __init__(self, df, char_list, n, weight='ew'):
		self.df = df # individual asset pd.DataFrame
		self.char_list = char_list # all characteristics of interest
		self.n = n      # unisort No. of buckets
		self.w = weight # ew: 'ew' or vw: 'lag_me' or 'log_me'

	def describe(self):
		print(self.df.shape)
		print(self.char_list)

	def sorted_portfolio_on_char(self, char):
		p=port(self.df, char, self.char_list, self.n, self.w)
		p.update_all()
		return p

	# def update_all(self):
	# 	print('#'*10, '\n', 'Start updating the cross-section data')
	# 	start = time.time()
	# 	for char in tqdm(self.char_list):
	# 		self.sorted_portfolio_on_char(char)
	# 		expression = "self.port_%s = self.sorted_portfolio_on_char(char)"%(char,char)
	# 		exec(expression)
	# 	end = time.time()
	# 	print('total time (s)= ' + str(end-start))
	# 	print('#'*10, '\n', 'Finish the cross-section data','\n','#'*10)

			
	def update_all(self, parallel=True):
		print('#'*10, '\n', 'Start updating the cross-section data')
		start = time.time()
		if not parallel:
			for char in tqdm(self.char_list):
				self.sorted_portfolio_on_char(char)
				expression = "self.port_%s = self.sorted_portfolio_on_char(char)"%(char)
				exec(expression)
		else:
			# nProcess = min(len(self.char_list),mp.cpu_count()-1)
			nProcess = 10
			print('Number of Process: %s'%(nProcess))
			p = mp.Pool(processes = nProcess)
			result = p.map_async(self.sorted_portfolio_on_char, self.char_list)
			p.close()
			p.join()
			for c in range(len(self.char_list)):
				sp = result.get()[c]
				sp.df = self.df
				ch = sp.char
				expression = "self.port_%s = sp"%(ch)
				exec(expression)
			del sp, ch, result
		end = time.time()
		print('total time (s)= ' + str(end-start))
		print('#'*10, '\n', 'Finish the cross-section data','\n','#'*10)