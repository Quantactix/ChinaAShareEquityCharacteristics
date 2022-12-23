import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class port:
	'''
	#Input:

	$df$: individual asset DataFrame
	$char$: characteristc for sorting
	$char_list$: all the characteristcs (features)
	$n$: the number of portfolios in the sorting procedure
	
	# Description
	port is a class,
	which do univariate sorting on the $char$ into $n$ portfolios.
	Also, the returns, number of assets, and the characteristics listed in $char_list$ are calculated for each portfolio.
	
	High-minus-low factor is provided, along with the ret, chars.
	'''
	def __init__(self, df, char, char_list=None, n=None, weight='ew'):
		self.df = df
		self.char = char
		self.char_list = char_list
		self.n = n
		# self.ret = None
		# self.count = None
		self.w = weight

	### Auxiliary Functions ###
	def describe(self):
		print(self.char)

	def plot_return(self):
		self.uni_ret.cumsum.plot()

	def plot_count(self):
		self.count.plot()

	def show_sharpe_ratio(self):
		uni_ls_sr = self.uni_ls.mean()/self.uni_ls.std()*np.sqrt(len(self.uni_ls))
		bi_ls_sr = self.bi_ls.mean()/self.bi_ls.std()*np.sqrt(len(self.bi_ls))
		return {'uni_ls_sr':uni_ls_sr, 'bi_ls_sr':bi_ls_sr}

	def value_weight_average(self, v, g, w):
		'''
		v,g,w are str, columns of the self.df
		v: variable of interest, 'ret'
		g: grouping columns, ['date','port_uni_bm']
		w: weight column: 'lag_me' or 'log_me'
		'''
		self.df['_weighted_v'] = self.df[v]*self.df[w]
		self.df['_weighted_notnull'] = pd.notnull(self.df[v])*self.df[w]
		g = self.df.groupby(g)
		vw = (g['_weighted_v'].sum() / g['_weighted_notnull'].sum()).unstack()
		del self.df['_weighted_v'], self.df['_weighted_notnull']
		return (vw)
	### END Auxiliary Functions ###
	
	### Main Functions ###
	def unisort(self):
		'''
		n=5, quintile portfolios, 0,1,2,3,4
		n=10, decile portfolios, 0,1,2,3,4,5,6,7,8,9
		'''
		rank_c = 'rank_'+self.char
		
		if self.n==10:
			tmp = np.where(self.df[rank_c]>=-0.8,1,0) \
			    + np.where(self.df[rank_c]>=-0.6,1,0) \
			    + np.where(self.df[rank_c]>=-0.4,1,0) \
			    + np.where(self.df[rank_c]>=-0.2,1,0) \
			    + np.where(self.df[rank_c]>= 0, 1,0)  \
			    + np.where(self.df[rank_c]>= 0.2,1,0) \
			    + np.where(self.df[rank_c]>= 0.4,1,0) \
			    + np.where(self.df[rank_c]>= 0.6,1,0) \
			    + np.where(self.df[rank_c]>= 0.8,1,0)
			self.df['port_uni_'+self.char] = [str(i) for i in tmp]

		if self.n==5:
			tmp = np.where(self.df[rank_c]>=-0.6,1,0) \
			    + np.where(self.df[rank_c]>=-0.2,1,0) \
			    + np.where(self.df[rank_c]>= 0.2,1,0) \
			    + np.where(self.df[rank_c]>= 0.6,1,0)
			self.df['port_uni_'+self.char] = [str(i) for i in tmp]

		if self.n==4:
			tmp = np.where(self.df[rank_c]>=-0.5,1,0) \
			    + np.where(self.df[rank_c]>=0,1,0) \
			    + np.where(self.df[rank_c]>= 0.5,1,0)
			self.df['port_uni_'+self.char] = [str(i) for i in tmp]

	def bisort(self):
		'''
		ME2 x CHAR3
		'''
		char = self.char
		self.df['bucket_char'] = \
			np.where(self.df['rank_'+char]<=-0.3333,
			char+'1',
			np.where(self.df['rank_'+char]<=0.3333,char+'2',char+'3'))
		self.df['bucket_me'] = \
			np.where(self.df['rank_me']<=0,
			'me1',
			'me2')
		self.df['port_bi_'+self.char] = self.df['bucket_me']+self.df['bucket_char']
		del self.df['bucket_char'], self.df['bucket_me']

	def get_port_attribute(self):

		self.uni_count = self.df.groupby(['date','port_uni_'+self.char]).count()['ret'].unstack()
		if not self.char=='me':
			self.bi_count = self.df.groupby(['date','port_bi_'+self.char]).count()['ret'].unstack()

		# equal weight
		if self.w == 'ew':
			# print('equal weight returns/characteristics')
			# unisort portfolio
			self.uni_ret = self.df.groupby(['date','port_uni_'+self.char]).mean()['ret'].unstack()
			for ch in self.char_list:
				expression = "self.uni_%s = self.df.groupby(['date','port_uni_'+self.char]).mean()['rank_%s'].unstack()"%(ch,ch)
				exec(expression)
			if not self.char=='me':
				# bisort portfolio
				self.bi_ret = self.df.groupby(['date','port_bi_'+self.char]).mean()['ret'].unstack()
				for ch in self.char_list:
					expression = "self.bi_%s = self.df.groupby(['date','port_bi_'+self.char]).mean()['rank_%s'].unstack()"%(ch,ch)
					exec(expression)
		# value weight
		elif (self.w=='lag_me') or (self.w=='log_me'):
			# print('vale %s weight returns/characteristics'%(self.w))
			# unisort portfolio
			self.uni_ret = self.value_weight_average('ret',['date','port_uni_'+self.char],self.w)
			for ch in self.char_list:
				expression = "self.uni_%s = self.value_weight_average('rank_%s',['date','port_uni_'+self.char],self.w)"%(ch,ch)
				exec(expression)
			if not self.char=='me':
				# bisort portfolio
				self.bi_ret = self.value_weight_average('ret',['date','port_bi_'+self.char],self.w)
				for ch in self.char_list:
					expression = "self.bi_%s = self.value_weight_average('rank_%s',['date','port_bi_'+self.char],self.w)"%(ch,ch)
					exec(expression)
		else:
			print('Invalid weight %s'%(self.ws))
			print('This is Error.')
			True==False

	def long_short_portfolio(self):
		'''
		Quitle Porfolio LS Factor
		Decile Portfolio LS Factor

		2x3 Factor
		HML = 1/2 (Small Value + Big Value) - 1/2 (Small Growth + Big Growth).
		http://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/f-f_factors.html
		'''
		if self.uni_ret[str(self.n-1)].mean() >= self.uni_ret['0'].mean():
			self.ls_sign = 'Long High'
			self.uni_ls = self.uni_ret[str(self.n-1)] - self.uni_ret['0']
			if not self.char=='me':
				self.bi_ls = 1/2*(self.bi_ret['me1'+self.char+'3']+self.bi_ret['me2'+self.char+'3']) \
					    - 1/2*(self.bi_ret['me1'+self.char+'1']+self.bi_ret['me2'+self.char+'1'])
		else:
			self.ls_sign = 'Long Low'
			self.uni_ls = self.uni_ret['0'] - self.uni_ret[str(self.n-1)]
			if not self.char=='me':
				self.bi_ls = 1/2*(self.bi_ret['me1'+self.char+'1']+self.bi_ret['me2'+self.char+'1']) \
					    - 1/2*(self.bi_ret['me1'+self.char+'3']+self.bi_ret['me2'+self.char+'3'])
		print(self.ls_sign)
		# Sharpe ratio
		self.uni_ls_sr = self.uni_ls.mean()/self.uni_ls.std()*np.sqrt(12)
		if not self.char=='me':
			self.bi_ls_sr = self.bi_ls.mean()/self.bi_ls.std()*np.sqrt(12)
	### END Main Functions ###

	def update_all(self):
		print('\n ### Start working on %s ###'%(self.char))
		self.unisort()
		self.bisort()
		self.get_port_attribute()
		self.long_short_portfolio()
		print('### Finish working on %s ### \n'%(self.char))



