#!/usr/bin/env python

import os,sys,itertools,inspect,copy,json,pickle,importlib,timeit,time,glob
from natsort import natsorted

import numpy as np
import scipy as sp
import pandas as pd
import scipy.stats,scipy.signal,scipy.cluster,subprocess

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


fontsize = 130
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.size'] = fontsize
matplotlib.rcParams['font.weight'] = 'bold'
matplotlib.rcParams['axes.labelweight'] = 'bold'
matplotlib.rcParams['axes.linewidth'] = 12






# Error Analysis Postprocessing functions

def position(site,n,d,dtype=np.int32):
	# Return position coordinates in d-dimensional n^d lattice 
	# from given linear site position in 1d N^d length array
	# i.e) [int(site/(n**(i))) % n for i in range(d)]
	n_i = np.power(n,np.arange(d,dtype=dtype))

	isint = isinstance(site,int)

	if isint:
		site = np.array([site])
	position = np.mod(((site[:,None]/n_i)).astype(dtype),n)
	if isint:
		return position[0]
	else:
		return position

def site(position,n,d,dtype=np.int32):
	# Return linear site position in 1d N^d length array 
	# from given position coordinates in d-dimensional n^d lattice
	# i.e) sum(position[i]*n**i for i in range(d))
	
	n_i = np.power(n,np.arange(d,dtype=dtype))

	is1d = isinstance(position,(int,list,tuple)) or position.ndim < 2

	if is1d:
		position = np.atleast_2d(position)
	
	site = (np.dot(position,n_i)).astype(dtype)

	if is1d:
		return site[0]
	else:
		return site


# Check if number
def isnumber(s):
	try:
		s = float(s)
		return True
	except:
		try:
			s = int(s)
			return True
		except:
			return False



# Replace text
def replacements(string):
	# string = string.replace(r'\textrm',r'\small\textnormal')
	string = string.replace(r'\textrm{B}',r'B')

	return string



# Put Exponent number into Scientific Notation string
def scinotation(number,decimals=2,order=2,base=10,zero=True,usetex=True,scilimits=[-1,1]):
	if not isnumber(number):
		return number
	number = int(number) if int(number) == float(number) else float(number)

	maxnumber = base**order
	if number > maxnumber:
		number = number/maxnumber
		if int(number) == number:
			number = int(number)
		string = str(number)
	
	if zero and number == 0:
		string = '%d'%(number)
	
	elif isinstance(number,int):
		string = str(number)
		if usetex:
			string = r'\textrm{%s}'%(string)
	
	elif isinstance(number,float):		
		string = '%0.*e'%(decimals,number)
		string = string.split('e')
		basechange = np.log(10)/np.log(base)
		basechange = int(basechange) if int(basechange) == basechange else basechange
		flt = string[0]
		exp = str(int(string[1])*basechange)
		if usetex:
			if int(exp) in range(*scilimits):
				flt = '%0.*f'%(decimals,float(flt)/(base**(-int(exp))))
				string = r'%s'%(flt)
			else:
				string = r'%s%s'%(flt,r'\cdot %d^{%s}'%(base,exp) if exp!= '0' else '')
	if usetex:
		string = r'$%s$'%(string.replace('$',''))
	else:
		string = string.replace('$','')
	string = string.replace('$','')
	return string



def combinations(p,n,unique=False):
	''' 
	Get all combinations of p number of non-negative integers that sum up to at most n
	Args:
		p (int): Number of integers
		n (int): Maximum sum of integers
		unique (bool): Return unique combinations of integers and q = choose(p+n,n) else q = (p^(n+1)-1)/(p-1)
	Returns:
		combinations (ndarray): All combinations of integers of shape (q,p)
	'''
	combinations = []
	iterable = range(p)
	for i in range(n+1):
		combos = list((tuple((j.count(k) for k in iterable)) for j in itertools.product(iterable,repeat=i)))
		if unique:
			combos = sorted(set(combos),key=lambda i:combos.index(i))
		combinations.extend(combos)
		
	combinations = np.vstack(combinations)
	return combinations



class ScalarFormatterForceFormat(matplotlib.ticker.ScalarFormatter):
	def _set_format(self,vmin,vmax):  # Override function that finds format to use.
		format = "%0.1f"  # Give format here 



def logfit(x,y):

	y = y.copy()
	y[y==0] = 1

	logx = np.log(x)
	logy = np.log(y)

	x_pred = x
	logx_pred = np.log(x_pred)
	logX = np.concatenate((np.ones(logx[:,None].shape),logx[:,None]),axis=1)
	logY = logy
	logcoef_ = np.linalg.pinv(logX).dot(logY)
	logy_pred = (logX).dot(logcoef_)

	y_pred = np.exp(logy_pred)    

	return x_pred,y_pred,logcoef_


# Shapes
def polygon(poly,coords,shape,text,ax,**props):
	
	if poly in ['rectangle','square']:
		patch = matplotlib.patches.Rectangle(coords,*shape,**props)
	elif poly in ['triangle','polygon']:
		patch = matplotlib.patches.Polygon(coords,**props)
	ax.add_patch(patch)

	for t in text:
		ax.text(**t)
	return



def triangle(x,y,slope,scale,offsets,ax):
	base = 10
	poly = 'triangle'
	coords = np.array([
		[x,y],
		[x*(base**(scale)),y],
		[x*(base**(scale)),y*(base**(scale*slope))]
		])
	shape = None
	polyprops = {'facecolor':plt.get_cmap('tab10')(7),'edgecolor':'black','linewidth':4,'closed':True}
	text = [
		{'s':r'$1$','x':x*(base**(scale*offsets[0][0])),'y':y*(base**(scale*offsets[0][1])),'fontsize':fontsize-20},
		{'s':r'$%d$'%(slope),'x':x*(base**(scale*(1-offsets[1][0]))),'y':y*(base**(scale*slope*(1-offsets[1][1]))),'fontsize':fontsize-20}
		]
	polygon(poly,coords,shape,text,ax,**polyprops)
	
	return

# Error Scaling Plotting

def scaling(paths=['analysis__*.pickle']):

	paths = [subpath for path in paths for subpath in glob.glob(path)]
	data = {}

	for path in paths[:]:
		try:
			with open(path,'rb') as fpath:
				data[path] = pickle.load(fpath)
		except Exception as e:
			paths.remove(path)
			pass
	Npaths = len(paths)

	if Npaths == 0:
		return

	texify = {}
	texify.update({
				'BOR':r'\eta_{\textrm{B}}',
				'FUELTEMP':r'T_{\textrm{F}}',
				'MODDENS':r'\rho_{\textrm{M}}',
				'MODTEMP':r'T_{\textrm{M}}',
				'RODFRAC':r'\eta_{\textrm{R}}',
				'BURNUP':r'f_{\textrm{brn}}',
				'u': r'u',
				})
	texify.update({
				# **{'analysis__XS%s_%d_%d_%d'%(s,g,0 if s not in ['S'] else k,i):r'\Sigma_{%s_{%s}}'%(
				# 	s.lower(),'%s%s'%(str(g+1),'' if s not in ['S'] else str(k+1))) 
				#    for s in ['TR','F','RM'] for g in [0,1] for k in [0,1]
				#    for i in range(10)},
				**{'analysis__XS%s_%d_%d_%d'%(s,g,0 if s not in ['S'] else k,i):r'\Sigma_{k}(x)~\textrm{Taylor Series Scaling}'
				   for s in ['TR','F','RM'] for g in [0,1] for k in [0,1]
				   for i in range(10)},			   
				**{'analysis__u_%d_loss'%(i):r'u_{k}(x)~\textrm{Taylor Series Scaling}' for i in range(10)},
				**{'analysis__u_n%d_K%d_k%d'%(i,j,k):r'u_{k}(x)~\textrm{Taylor Series Scaling}' for i,j,k in [(i,j,k) for i in range(2**11) for j in range(10) for k in range(10)]}
				})
	texify.update({
				r'{x_0^{}}{^{}}':r'x',
				r'{^{}}{x_1^{}}':r'y',
				r'{x_0^{2}}{^{}}':r'x^{2}',
				r'{^{}}{x_1^{2}}':r'y^{2}',
				r'{x_0^{3}}{^{}}':r'x^{3}',
				r'{^{}}{x_1^{3}}':r'y^{3}',								
			})


	# Data
	paths = list(sorted(paths,key=lambda path: min([d['order'] for i in data[path] for d in data[path][i]])))
	directory = os.path.dirname(os.path.commonpath(paths))
	keys = [key for path in paths for key in data[path]]
	keys = natsorted(sorted(keys,key=lambda key: (min([d['order'] for path in paths for d in data[path].get(key,[]) if key in data[path]]),
										min([d['K'] for path in paths for d in data[path].get(key,[]) if key in data[path]]))))
	orders = list(set([d['order'] for key in keys for path in paths for d in data[path].get(key,[]) if key in data[path]]))
	K = list(set([d['K'] for key in keys for path in paths for d in data[path].get(key,[]) if key in data[path]]))
	N = list(set([d['n'] for key in keys for path in paths for d in data[path].get(key,[]) if key in data[path]]))
	P = list(set([d['p'] for key in keys for path in paths for d in data[path].get(key,[]) if key in data[path]]))


	types = ['global','local']
	names = ['error','derivativeerror','gamma']
	names = ['error','derivativeerror']#,'gamma']
	names = {name: None for name in names}
	isname = {name: False for name in names}
	for name in names:
		names[name] = [d.get(name) for key in keys for path in paths for d in data[path].get(key,[]) if key in data[path]]
		isname[name] = all([x is not None for x in names[name]])


	names = [name for name in names if isname[name]]
	nfigs = len([name for name in isname if isname[name]])
	nrows = 1
	ncols = 1

	figs,axes = {},{}
	for typed in types:
		figs[typed],axes[typed] = {},{}
		for name in names:
			figs[typed][name],axes[typed][name] = plt.subplots(nrows,ncols)     
			axes[typed][name] = np.atleast_2d(axes[typed][name])
			axes[typed][name] = axes[typed][name].T if axes[typed][name].shape[0]==1 and nrows>ncols else axes[typed][name]

		
	# fig.set_size_inches(Npaths*40,Npaths*24);
	figsize = 55
	for typed in types:
		for name in names:
			figs[typed][name].set_size_inches(figsize,figsize);

	colors = lambda i: plt.get_cmap('tab10')(i)
	# markers = lambda i: ['o','o','^'][i%3]
	markers = lambda i: ['o'][i%1]

	plotprops = {'linewidth':25,'markersize':80,'alpha':0.75}
	legendprops = lambda i: {"prop": {"size": 90},'markerscale':0.8,'ncol':2,'loc':(1.05,-0.2),'framealpha':1}


	for name in names:
		for icol in range(ncols):
			for irow in range(nrows):
				for key in keys:

					path = [path for path in paths if key in data[path]][0]
					ipath = paths.index(path)

					# Parameters
					order = [d['order'] for path in paths for d in data[path].get(key,[]) if key in data[path]][0]
					q = [d['q'] for path in paths for d in data[path].get(key,[]) if key in data[path]][0]
					r = [d['r'] for path in paths for d in data[path].get(key,[]) if key in data[path]][0]
					operator = [d['operator'] for path in paths for d in data[path].get(key,[]) if key in data[path]][0]
					modelstring = [d['modelstring'] for path in paths for d in data[path].get(key,[]) if key in data[path]][0]
					manifold = [d['manifold'] for path in paths for d in data[path].get(key,[]) if key in data[path]][0]
					n = np.array([d['n'] for path in paths for d in data[path].get(key,[]) if key in data[path]])
					L = np.array([d['L'] for path in paths for d in data[path].get(key,[]) if key in data[path]])
					h = np.array([d['h'] for path in paths for d in data[path].get(key,[]) if key in data[path]])

					alpha = np.array([d['alpha'] for path in paths for d in data[path].get(key,[]) if key in data[path]])
					p = np.array([d['p'] for path in paths for d in data[path].get(key,[]) if key in data[path]])
					k = [d['K'] for path in paths for d in data[path].get(key,[]) if key in data[path]]


					if name == 'error':


						y0 = 1e-2

						# Data				
						x = h/L
						y = np.array([d[name] for path in paths for d in data[path].get(key,[]) if key in data[path]])

						offset = 1#/y[0]
						y *= offset

						# Fit Data
						x_pred,y_pred,logcoef_ = logfit(x,y)

						offset_pred = 1#/y_pred[0]
						y_pred *= offset_pred

						# Calculation
						# x_calc = h/L
						# y_calc = alpha[:,2]/np.sqrt(3)*(1+ (3/2)*(h/L)**2)*h*L

						# offset_calc = 1#/y_calc[0]
						# y_calc *= offset_calc

						# Model
						# x_model = h/L
						# y_model = np.array([d['modelerror'] for path in paths for d in data[path].get(key,[]) if key in data[path]])
						# y_model = np.array([d['modelerror'] for d in data[path][key]])

						# offset_model = 1#/y_model[0]
						# y_model *= offset_model


						# Fit Model
						# x_pred_model,y_pred_model,logcoef__model = logfit(x,y)

						# offset_pred_model = 1#/y_pred[0]
						# y_pred_model *= offset_pred_model


						# Local data
						if np.all(p == 1):
							points = {
								'0': lambda n,d: site([0],n,d),
								# 'L/4': lambda n,d: site([n//4-1],n,d),
								'L/2': lambda n,d: site([n//3-1],n,d),
								# '3L/4': lambda n,d: site([(3*(n//4))-1],n,d),
								'L': lambda n,d: site([n-1],n,d),
							}
							# points = {}
						elif np.all(p == 2):
							points = {
								'0,0': lambda n,d: site([0,0],n,d),
								# 'L/2,0': lambda n,d: site([n//2-1,0],n,d),
								# '0,L/2': lambda n,d: site([0,n//2-1],n,d),
								'L,0': lambda n,d: site([n-1,0],n,d),
								'L/2,L/2': lambda n,d: site([n//2-1,n//2-1],n,d),
								'0,L': lambda n,d: site([0,n-1],n,d),
								'L,L': lambda n,d: site([n-1,n-1],n,d),
							}
						else:
							points = {
								','.join([{'0':'0','n//2-1':'L/2','n-1':'L'}[o] for o in point]): (lambda n,d,point=point: site([{'0':0,'n//2-1':n//2-1,'n-1':n-1}[o] for o in point],n,d)) for point in itertools.product(['0','n//2-1','n-1'],repeat=p[0])
							}


						_x_local = h/L
						_y_local = [d['localerror'] for path in paths for d in data[path].get(key,[]) if key in data[path]]
						x_local = {}
						y_local = {}
						x_pred_local = {}
						y_pred_local = {}
						logcoef_pred_local = {}
						for point in points:
							x_local[point] = _x_local
							y_local[point] = np.abs([_y[points[point](_n,_d)] for _y,_n,_d in zip(_y_local,n,p)])
							x_pred_local[point],y_pred_local[point],logcoef_pred_local[point] = logfit(x_local[point],y_local[point])

						# Local model
						# _x_local_model = h/L
						# _y_local_model = [d['localmodelerror'] for path in paths for d in data[path].get(key,[]) if key in data[path]]
						# x_local_model = {}
						# y_local_model = {}
						# x_pred_local_model = {}
						# y_pred_local_model = {}
						# logcoef_pred_local_model = {}
						# for point in points:
						# 	x_local_model[point] = _x_local_model
						# 	y_local_model[point] = np.abs([_y[points[point](_n,_d)] for _y,_n,_d in zip(_y_local_model,n,p)])
						# 	x_pred_local_model[point],y_pred_local_model[point],logcoef_pred_local_model[point] = logfit(x_local_model[point],y_local_model[point])
					


						# Labels
						label = r'$||e_{%s}||_2 \sim O(%s~h^{%0.2f})$'%('',scinotation(np.exp(logcoef_[0])),logcoef_[1])   
						label_local = {point: r'$|e(%s)| \sim O(%s~h^{%0.2f})$'%(point,scinotation(np.exp(logcoef_pred_local[point][0])),logcoef_pred_local[point][1])   
										for point in points}

						typed = 'global'
						ax = axes[typed][name][irow][icol]
						ax.plot(x,y,color=colors(ipath),marker='s',linestyle='-',label=label,**plotprops,zorder=100)
						
						typed = 'local'
						ax = axes[typed][name][irow][icol]
						for i,point in enumerate(points):
							ax.plot(x_local[point],y_local[point],color=colors(ipath+1+i-1),marker=markers(k[0]),linestyle='-',label=label_local[point],**plotprops)


						# label = r'$||%s - (%s)||_2 \sim O(%s~h^{%0.2f})~\textrm{data}$'%(modelstring.replace('$','').replace('=','-'),'='.join(key.replace('$','').split('=')[1:]),scinotation(np.exp(logcoef_[0])),logcoef_[1])   
						# label_pred = r'$%s h^{%0.2f}$'%(scinotation(np.exp(logcoef_[0])),logcoef_[1])   
						# label_model = r'$||%s - (%s)||_{2} \sim O(%s~h^{%0.2f})~\textrm{model}$'%(modelstring.replace('$','').replace('=','-'),'='.join(key.replace('$','').split('=')[1:]),scinotation(np.exp(logcoef__model[0])),logcoef__model[1])   
						# label_pred_model = r'$%s h^{%0.2f}$'%(scinotation(np.exp(logcoef__model[0])),logcoef__model[1])   				
						# label_calc = r'$||%s - (%s)||_{2_{\textrm{calc}}}$'%(modelstring.replace('$','').replace('=','-'),'='.join(key.replace('$','').split('=')[1:]))
						# label_local = {point: r'$|%s - (%s)|(%s) \sim O(%s~h^{%0.2f})~ \textrm{data}$'%(modelstring.replace('$','').replace('=','-'),'='.join(key.replace('$','').split('=')[1:]),point,scinotation(np.exp(logcoef_pred_local[point][0])),logcoef_pred_local[point][1])   
						# 				for point in points}
						
						# label_local_model = {point: r'$|%s - (%s)|(%s) \sim O(%s~h^{%0.2f})~ \textrm{model}$'%(modelstring.replace('$','').replace('=','-'),'='.join(key.replace('$','').split('=')[1:]),point,scinotation(np.exp(logcoef_pred_local_model[point][0])),logcoef_pred_local_model[point][1])   
						# 				for point in points}


						# Plots
						# ax.plot(x,y,'-s',color=colors(ipath),marker=markers(k[0]),label=label,**plotprops)
						# ax.plot(x,y,'-s',color=colors(ipath),marker='s',label=label,**plotprops,zorder=100)
						# ax.plot(x_model,y_model,color='k',marker='+',label=label_model,**plotprops)
						# ax.plot(x_pred_model,y_pred_model,'--',color='k',linewidth=12,markersize=5,zorder=10,alpha=0.9)				
						# ax.plot(x_calc,y_calc,color='k',marker='o',label=label_calc,**plotprops)
						# ax.plot(x_pred,y_pred,'--',color=colors(ipath),linewidth=12,markersize=5,zorder=10,alpha=0.9)

						# for i,point in enumerate(points):
						# 	ax.plot(x_local[point],y_local[point],'-s',color=colors(ipath+1+i),marker=markers(k[0]),label=label_local[point],**plotprops)
							# ax.plot(x_local_model[point],y_local_model[point],color=colors(ipath+1+i),marker='+',label=label_local_model[point],**plotprops)



					elif name == 'derivativeerror':


						# try:
						# for j,icoef in enumerate(np.sort(np.random.choice(np.arange(1,q),size=3,replace=False))):
						if np.all(p==1):
							icoefs = np.arange(1,q)
						elif np.all(p==2):
							icoefs = [p[0]-1,p[0]+3,((1+p[0])*(2+p[0]))//2+0]
						else:
							icoefs = np.arange(1,q)
						for j,icoef in enumerate(icoefs):

							y0 = 1e-2

							# Data				
							x = h/L
							y = np.array([d[name][icoef] for path in paths for d in data[path].get(key,[]) if key in data[path]])

							offset = 1#/y[0]
							y *= offset

							# Fit Data
							x_pred,y_pred,logcoef_ = logfit(x,y)
							offset_pred = 1#/y_pred[0]
							y_pred *= offset_pred



							# Local data
							if np.all(p == 1):
								points = {
									'0': lambda n,d: site([0],n,d),
									# 'L/4': lambda n,d: site([n//4-1],n,d),
									# 'L/2': lambda n,d: site([n//3-1],n,d),
									# '3L/4': lambda n,d: site([(3*(n//4))-1],n,d),
									'L': lambda n,d: site([n-1],n,d),
								}
								# points = {}
							elif np.all(p == 2):
								points = {
									'0,0': lambda n,d: site([0,0],n,d),
									# 'L/2,0': lambda n,d: site([n//2-1,0],n,d),
									# '0,L/2': lambda n,d: site([0,n//2-1],n,d),
									'L,0': lambda n,d: site([n-1,0],n,d),
									# 'L/2,L/2': lambda n,d: site([n//2-1,n//2-1],n,d),
									# '0,L': lambda n,d: site([0,n-1],n,d),
									'L,L': lambda n,d: site([n-1,n-1],n,d),
								}
							else:
								points = {
									','.join([{'0':'0','n//2-1':'L/2','n-1':'L'}[o] for o in point]): (lambda n,d,point=point: site([{'0':0,'n//2-1':n//2-1,'n-1':n-1}[o] for o in point],n,d)) for point in itertools.product(['0','n//2-1','n-1'],repeat=p[0])
								}														

							_x_local = h/L
							_y_local = [d['localderivativeerror'][icoef] for path in paths for d in data[path].get(key,[]) if key in data[path]]


							x_local = {}
							y_local = {}
							x_pred_local = {}
							y_pred_local = {}
							logcoef_pred_local = {}


							for point in points:
								x_local[point] = _x_local
								y_local[point] = np.abs([_y[points[point](_n,_d)] for _y,_n,_d in zip(_y_local,n,p)])
								x_pred_local[point],y_pred_local[point],logcoef_pred_local[point] = logfit(x_local[point],y_local[point])



							# Labels
							if np.all(p == 1):
								_label = manifold[icoef]
								_label = _label[-3:-2]
								if _label == "{":
									_label = '1'
							elif np.all(p == 2):
								_label = manifold[icoef]
								if _label.startswith('{^{}}'):
									_label = _label[-3:-2]
									if _label == "{":
										_label = '1'
									_label = "0"+_label
								elif _label.endswith('{^{}}'):
									_label = _label[6:7]
									if _label == "}":
										_label = '1'
									_label = _label+"0"
								else:
									_label = _label[6:7]+_label[-3:-2]
									if _label.startswith('}'):
										_label = '1'+_label[1:]
									elif _label.endswith('{'):
										_label = _label[:-1]+'1'										

							else:
								_label = manifold[icoef]

							label = r'$||\varepsilon_{%s}||_2 \sim O(%s~h^{%0.2f})$'%(texify.get(_label,_label),scinotation(np.exp(logcoef_[0])),logcoef_[1])   
							label_local = {point: r'$|\varepsilon_{%s}(%s)| \sim O(%s~h^{%0.2f})$'%(texify.get(_label,_label),point,scinotation(np.exp(logcoef_pred_local[point][0])),logcoef_pred_local[point][1])   
											for point in points}


							# Plots
							typed = 'global'
							ax = axes[typed][name][irow][icol]
							ax.plot(x,y,marker='s',linestyle='-',label=label,**plotprops)

							typed = 'local'
							ax = axes[typed][name][irow][icol]
							for i,point in enumerate(np.sort(np.random.choice(list(points),size=p[0]+1,replace=False))):
								ax.plot(x_local[point],y_local[point],color=colors(ipath+j*(p[0]+1)+1+i-1),marker=markers(k[0]),linestyle='-',label=label_local[point],**plotprops)



							# Calculation
							# x_calc = h/L
							# y_calc = alpha[:,2]/np.sqrt(3)*(1+ (3/2)*(h/L)**2)*h*L

							# offset_calc = 1#/y_calc[0]
							# y_calc *= offset_calc

							# # Model
							# x_model = h/L
							# y_model = np.array([d['modelerror'] for path in paths for d in data[path].get(key,[]) if key in data[path]])
							# # y_model = np.array([d['modelerror'] for d in data[path][key]])

							# offset_model = 1#/y_model[0]
							# y_model *= offset_model


							# # Fit Model
							# x_pred_model,y_pred_model,logcoef__model = logfit(x,y)

							# offset_pred_model = 1#/y_pred[0]
							# y_pred_model *= offset_pred_model

							# # Local model
							# _x_local_model = h/L
							# _y_local_model = [d['localmodelerror'] for path in paths for d in data[path].get(key,[]) if key in data[path]]
							# x_local_model = {}
							# y_local_model = {}
							# x_pred_local_model = {}
							# y_pred_local_model = {}
							# logcoef_pred_local_model = {}
							# for point in points:
							# 	x_local_model[point] = _x_local_model
							# 	y_local_model[point] = np.abs([_y[points[point](_n,_d)] for _y,_n,_d in zip(_y_local_model,n,p)])
							# 	x_pred_local_model[point],y_pred_local_model[point],logcoef_pred_local_model[point] = logfit(x_local_model[point],y_local_model[point])


						


							# Labels
							# label = r'$||\frac{\delta %s}{\delta x} - \frac{\delta %s}{\delta x}||_2 \sim O(%s~h^{%0.2f})~\textrm{data}$'%(modelstring.replace('$','').replace('=','-'),'='.join(key.replace('$','').split('=')[1:]),scinotation(np.exp(logcoef_[0])),logcoef_[1])   
							# label = r'$||\varepsilon_{%d %s}||_2 \sim O(%s~h^{%0.2f})$'%(icoef,replacements(key),scinotation(np.exp(logcoef_[0])),logcoef_[1])   
							# label = r'$||\varepsilon_{%d}||_2 \sim O(%s~h^{%0.2f})$'%(icoef,scinotation(np.exp(logcoef_[0])),logcoef_[1])   
							# label_pred = r'$%s h^{%0.2f}$'%(scinotation(np.exp(logcoef_[0])),logcoef_[1])   
							# label_model = r'$||%s - (%s)||_{2} \sim O(%s~h^{%0.2f})~\textrm{model}$'%(modelstring.replace('$','').replace('=','-'),'='.join(key.replace('$','').split('=')[1:]),scinotation(np.exp(logcoef__model[0])),logcoef__model[1])   
							# label_pred_model = r'$%s h^{%0.2f}$'%(scinotation(np.exp(logcoef__model[0])),logcoef__model[1])   				
							# label_calc = r'$||%s - (%s)||_{2_{\textrm{calc}}}$'%(modelstring.replace('$','').replace('=','-'),'='.join(key.replace('$','').split('=')[1:]))
							# label_local = {point: r'$|\frac{\delta %s}{\delta x} - \frac{\delta %s}{\delta x}|(%s) \sim O(%s~h^{%0.2f})~ \textrm{data}$'%(modelstring.replace('$','').replace('=','-'),'='.join(key.replace('$','').split('=')[1:]),point,scinotation(np.exp(logcoef_pred_local[point][0])),logcoef_pred_local[point][1])   
							# 				for point in points}
							# label_local = {point: r'$|\varepsilon_{%d}(%s)| \sim O(%s~h^{%0.2f})$'%(icoef,point,scinotation(np.exp(logcoef_pred_local[point][0])),logcoef_pred_local[point][1])   
							# 				for point in points}
							# label_local_model = {point: r'$|%s - (%s)|(%s) \sim O(%s~h^{%0.2f})~ \textrm{model}$'%(modelstring.replace('$','').replace('=','-'),'='.join(key.replace('$','').split('=')[1:]),point,scinotation(np.exp(logcoef_pred_local_model[point][0])),logcoef_pred_local_model[point][1])   
							# 				for point in points}


							# Plots
							# ax.plot(x,y,'-s',color=colors(ipath),marker=markers(k[0]),label=label,**plotprops)
							# ax.plot(x,y,'-s',color=colors(ipath),marker='s',label=label,**plotprops)
							# ax.plot(x_model,y_model,color='k',marker='+',label=label_model,**plotprops)
							# ax.plot(x_pred_model,y_pred_model,'--',color='k',linewidth=12,markersize=5,zorder=10,alpha=0.9)				
							# ax.plot(x_calc,y_calc,color='k',marker='o',label=label_calc,**plotprops)
							# ax.plot(x_pred,y_pred,'--',color=colors(ipath),linewidth=12,markersize=5,zorder=10,alpha=0.9)

							# for i,point in enumerate(points):
							# 	ax.plot(x_local[point],y_local[point],'-s',color=colors(ipath+1+i),marker=markers(k[0]),label=label_local[point],**plotprops)
								# ax.plot(x_local_model[point],y_local_model[point],color=colors(ipath+1+i),marker='+',label=label_local_model[point],**plotprops)					

						# except Exception as e:
						# 	print('ERROR - ',e)
						# 	pass


					elif name == 'gamma':
						coef_ = [d[name] for path in paths for d in data[path].get(key,[]) if key in data[path]]
						for icoef in range(1,q):
							y = np.abs([c[icoef]-1 for c in coef_])[:]
							x = (h/L)[:]
							x_pred,y_pred,logcoef_ = logfit(x,y)
							
							# Labels
							if np.all(p == 1):
								_label = manifold[icoef]
								_label = _label[-3:-2]
								if _label == "{":
									_label = '1'
							elif np.all(p == 2):
								_label = manifold[icoef]
								if _label.startswith('{^{}}'):
									_label = _label[-3:-2]
									if _label == "{":
										_label = '1'
									_label = "0"+_label
								elif _label.endswith('{^{}}'):
									_label = _label[6:7]
									if _label == "}":
										_label = '1'
									_label = _label+"0"
								else:
									_label = _label[6:7]+_label[-3:-2]
									if _label.startswith('}'):
										_label = '1'+_label[1:]
									elif _label.endswith('{'):
										_label = _label[:-1]+'1'										

							else:
								_label = manifold[icoef]

							# label = r'$|\gamma_{%d_{%s}} - 1| \sim O(%s~h^{%0.2f})$'%(icoef,replacements(key),scinotation(np.exp(logcoef_[0])),logcoef_[1]) 
							label = r'$|\gamma_{%s} - 1| \sim O(%s~h^{%0.2f})$'%(texify.get(_label,_label),scinotation(np.exp(logcoef_[0])),logcoef_[1]) 

							typed = 'global'
							ax = axes[typed][name][irow][icol]
							ax.plot(x,y,color=colors(ipath+icoef),marker=markers(k[0]),linestyle='-',label=label,**plotprops)
						# ax.plot(x_pred,y_pred,'--',color='k',marker=None,**plotprops)

				for typed in types:
					ax = axes[typed][name][irow][icol]
					if icol == (0) and name == 'error':


						titlestring = (r'$\textrm{Function: } u(x) = \sum_{q=0}^{K=%d}\sum_{\mu = \{\mu_{1}\cdots\mu_{p=%d}\} : |\mu| = q}\alpha_{q_{\mu_{1}\cdots\mu_{p}}} x^{\mu_{1}\cdots\mu_{p}}$' +
									   r'\\~~\\' + 
									   r'$\textrm{Model: } u_{%d}(x|\tilde{x}) = \sum_{q=0}^{k=%d} \sum_{\mu = \{\mu_{1}\cdots\mu_{p=%d}\} : |\mu| = q} \gamma^{ \mu_{1}\cdots\mu_{p}}(\tilde{x})\frac{1}{q!}\frac{{\delta}^{q} u(\tilde{x})}{{\delta x}^{{\mu_{1}\cdots\mu_{p}}}}(x-\tilde{x})^{{\mu_{1}\cdots\mu_{p}}}$')

						# ax.set_title(titlestring%(k[0],p[0],order,order,p[0]),fontsize=fontsize+40,weight='bold',pad=600)
						ax.set_xlabel(r'\bf{$h$}',fontsize=fontsize+40,weight='bold')
						ax.set_ylabel(r'$\textrm{Error}$',fontsize=fontsize+40,weight='bold')
					elif icol == (0) and name == 'derivativeerror':
							ax.set_xlabel(r'\bf{$h$}',fontsize=fontsize+40,weight='bold')
							ax.set_ylabel(r'$\textrm{Error}$',fontsize=fontsize+40,weight='bold')
							# ax.set_ylabel(r'$\gamma_1 - 1$',fontsize=fontsize+40,weight='bold')
					elif icol == (0) and name == 'gamma':
							ax.set_xlabel(r'\bf{$h$}',fontsize=fontsize+40,weight='bold')
							ax.set_ylabel(r'$\textrm{Error}$',fontsize=fontsize+40,weight='bold')
							# ax.set_ylabel(r'$\gamma_1 - 1$',fontsize=fontsize+40,weight='bold')
					if name == 'error':
						pass
					# ax.tick_params(
					# 	axis='x',          # changes apply to the x-axis
					# 	which='both',      # both major and minor ticks are affected
					# 	bottom=False,      # ticks along the bottom edge are off
					# 	top=False,         # ticks along the top edge are off
					# 	labelbottom=False,
					# 	pad=10)
					if name == 'error':
						ax.set_xscale('log')
						ax.set_yscale('log')
					elif name == 'derivativeerror':
						ax.set_xscale('log')
						ax.set_yscale('log')
					elif name == 'gamma':
						ax.set_xscale('log')
						ax.set_yscale('log')
					

					if np.all(p==1):
						xticks = [1e-4,1e-3,1e-2,1e-1,5e-1]
						ylim = [1e-16,1e4]						
					elif np.all(p==2):
						xticks = [1e-5,1e-4,1e-3,1e-2,1e-1,5e-1]
						ylim = [1e-7,5e1]
					else:
						xticks = [1e-5,1e-4,1e-3,1e-2,1e-1,5e-1]
						ylim = [1e-7,5e1]
					if np.all(p==1) and name == 'error' and typed == 'global':					
						ax.set_xticks(xticks)
						ax.set_ylim(*ylim)
					elif np.all(p==1) and name == 'error' and typed == 'local':					
						ax.set_xticks(xticks)
						ax.set_ylim(*ylim)
					if np.all(p==1) and name == 'derivativeerror' and typed == 'global':					
						ax.set_xticks(xticks)
						ax.set_ylim(*ylim)
						# ax.set_xticks([1e-5,1e-4,1e-3,1e-2,1e-1,5e-1])
						# ax.set_ylim(**{'ymin':1e-7,'ymax':5e1})
					elif np.all(p==1) and name == 'derivativeerror' and typed == 'local':					
						ax.set_xticks(xticks)
						ax.set_ylim(*ylim)
					if np.all(p==2) and name == 'error' and typed == 'global':					
						ax.set_xticks(xticks)
						ax.set_ylim(*ylim)
					elif np.all(p==2) and name == 'error' and typed == 'local':					
						ax.set_xticks(xticks)
						ax.set_ylim(*ylim)
					if np.all(p==2) and name == 'derivativeerror' and typed == 'global':					
						ax.set_xticks(xticks)
						ax.set_ylim(*ylim)
						# ax.set_xticks([1e-5,1e-4,1e-3,1e-2,1e-1,5e-1])
						# ax.set_ylim(**{'ymin':1e-7,'ymax':5e1})
					elif np.all(p==2) and name == 'derivativeerror' and typed == 'local':					
						ax.set_xticks(xticks)
						ax.set_ylim(*ylim)
						# ax.set_xticks([1e-5,1e-4,1e-3,1e-2,1e-1,5e-1])
						# ax.set_ylim(**{'ymin':1e-7,'ymax':5e1})						

					# xformatter = matplotlib.ticker.ScalarFormatter()
					# yformatter = matplotlib.ticker.ScalarFormatter()
					# ax.xaxis.set_major_formatter(xformatter)
					# ax.yaxis.set_major_formatter(yformatter)
					# ax.ticklabel_format(**{"axis":"y","style":"sci","scilimits":[-1,1]})    
					# ax.ticklabel_format(**{"axis":"x","style":"sci","scilimits":[1,2]})    

					# ax.set_xticks([],minor=True)
		  	# 	    ax.set_xticklabels([2,4,r'$6 \cdot 10^{-1}$'])
					# ax.tick_params(axis='x', which='minor', labelsize=fontsize+40,length=20,pad=50)
					# ax.tick_params(axis='y', which='minor', labelsize=fontsize+40,length=20,pad=50)
					ax.tick_params(axis='x', which='major', labelsize=fontsize+50,length=30,pad=50)
					ax.tick_params(axis='y', which='major', labelsize=fontsize+50,length=30,pad=50)

					# ax.tick_params(axis='x', which='both', labelsize=fontsize+40,length=20,pad=50)
					# ax.tick_params(axis='y', which='both', labelsize=fontsize+40,length=20,pad=50)
					# ax.set_xlim(**{'xmin':1e-5,'xmax':1e0})
					# ax.set_xticks([1e-2,5e-2,1e-1])

					# locmaj = matplotlib.ticker.LogLocator(base=10.0, subs=(1.0, ))
					# ax.xaxis.set_major_locator(locmaj)

					# locmin = matplotlib.ticker.LogLocator(base=10.0, subs=(.1,))
					# ax.xaxis.set_minor_locator(locmin)
					# ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())


					handles,labels = getattr(ax,'get_legend_handles_labels')()
					if len(handles) > 0 and len(labels) > 0:
						if name == 'error':
							if 'gamma' in names and typed in ['global']:
								if np.all(p==1):
									ax.legend(**{**legendprops(icol),**{'loc':(1.05,0.4)}})
								else:
									ax.legend(**{**legendprops(icol),**{'loc':(1.05,0.3)}})
							elif 'gamma' not in names and typed in ['global']:
								if np.all(p==1):
									# ax.legend(**{**legendprops(icol),**{'loc':(1.05,0.5),'ncol':1}})
									ax.legend(**{**legendprops(icol),**{'loc':'lower right','ncol':1}})																		
								else:
									# ax.legend(**{**legendprops(icol),**{'loc':(1.05,0.5),'ncol':1}})
									ax.legend(**{**legendprops(icol),**{'loc':'upper left','ncol':1}})
							elif 'gamma' not in names and typed in ['local']:
								if np.all(p==1):
									# ax.legend(**{**legendprops(icol),**{'loc':(1.05,0.5),'ncol':1}})
									ax.legend(**{**legendprops(icol),**{'loc':'lower right','ncol':1}})																		
								else:
									# ax.legend(**{**legendprops(icol),**{'loc':(1.05,0.3),'ncol':1}})
									ax.legend(**{**legendprops(icol),**{'loc':'upper left','ncol':1}})

						if name == 'derivativeerror':
							if 'gamma' in names:
								if np.all(p==1):
									ax.legend(**{**legendprops(icol),**{'loc':(1.05,-0.05),'ncol':(q-1)//2}})
								else:
									ax.legend(**{**legendprops(icol),**{'loc':(1.05,-0.25),'ncol':(q-1)//2}})
							else:
								if np.all(p==1):
									# ax.legend(**{**legendprops(icol),**{'loc':(1.05,0.1),'ncol':1}})
									ax.legend(**{**legendprops(icol),**{'loc':'lower right','ncol':1}})									
								else:
									# ax.legend(**{**legendprops(icol),**{'loc':(1.05,0.1),'ncol':1}})
									if typed in ['global']:
										ax.legend(**{**legendprops(icol),**{'loc':'upper left','ncol':1}})
									elif typed in ['local']:
										ax.legend(**{**legendprops(icol),**{'loc':'upper left','ncol':1}})

						elif name == 'gamma':
							if np.all(p==1):
								ax.legend(**{**legendprops(icol),**{'loc':(1.05,0.2),'ncol':1}})
							else:
								ax.legend(**{**legendprops(icol),**{'loc':(1.05,-0.2),'ncol':1}})


					ax.grid(True,which='major',linewidth=6,alpha=0.5)
					# ax.grid(True,which='major',linewidth=10,alpha=1)
					ax.set_axisbelow(True)


					if np.all(p==1) and name == 'error' and typed == 'global':
						a,b = 1.2e-1,3e0
						scale = 0.55
						offsets = [[0.65,0.15],[0.15,0.75]]
						base = 10
						offset = 1
						triangle(a,b*(base**(0*offset)),order+1,scale,offsets,ax)
					elif np.all(p==1) and name == 'error' and typed == 'local':
						a,b = 1.2e-1,3e0
						scale = 0.55
						offsets = [[0.65,0.15],[0.15,0.75]]						
						base = 10
						offset = 1
						triangle(a,b*(base**(0*offset)),order+1,scale,offsets,ax)
					elif np.all(p==1) and name == 'derivativeerror' and typed == 'global':
						a,b = 1.2e-1,3e-8
						scale = 0.55
						offsets = [[0.65,0.15],[0.15,0.75]]						
						base = 10
						offset = 1.5
						triangle(a,b*(base**(0*offset)),order+1,scale,offsets,ax)
						triangle(a,b*(base**(scale*(1*order + 1) + 1*offset)),order-1,scale,offsets,ax)
						triangle(a,b*(base**(scale*(2*order + 1 - 1) + 2*offset)),order-3,scale,offsets,ax)
					elif np.all(p==1) and name == 'derivativeerror' and typed == 'local':
						a,b = 1.2e-1,3e-5
						scale = 0.55
						offsets = [[0.65,0.15],[0.15,0.75]]						
						base = 10
						offset = 0.75
						triangle(a,b*(base**(0*offset)),order+1,scale,offsets,ax)
						triangle(a,b*(base**(scale*(1*order + 1) + 1*offset)),order-1,scale,offsets,ax)
						triangle(a,b*(base**(scale*(2*order + 1 - 1) + 2*offset)),order-3,scale,offsets,ax)
					
					if np.all(p==2) and name == 'error' and typed == 'global':
						a,b = 1.2e-1,1.5e-1
						scale = 0.5
						offsets = [[0.65,0.15],[0.225,0.6]]						
						base = 10
						offset = 1.5
						triangle(a,b*(base**(0*offset)),order+1,scale,offsets,ax)
					elif np.all(p==2) and name == 'error' and typed == 'local':
						a,b = 1.2e-1,1.5e-1
						scale = 0.5
						offsets = [[0.65,0.15],[0.225,0.6]]						
						base = 10
						offset = 1.5
						triangle(a,b*(base**(0*offset)),order+1,scale,offsets,ax)
					elif np.all(p==2) and name == 'derivativeerror' and typed == 'global':
						a,b = 1.2e-1,1.5e-5
						scale = 0.5
						offsets = [[0.65,0.15],[0.225,0.6]]	
						base = 10
						offset = 0.6
						triangle(a,b*(base**(0*offset)),order+1,scale,offsets,ax)
						triangle(a,b*(base**(scale*(1*order + 1) + 1*offset)),order,scale,offsets,ax)
						triangle(a,b*(base**(scale*(2*order + 1) + 2*offset)),order-1,scale,offsets,ax)
					elif np.all(p==2) and name == 'derivativeerror' and typed == 'local':
						a,b = 1.2e-1,1.5e-5
						scale = 0.5
						offsets = [[0.65,0.15],[0.225,0.6]]	
						base = 10
						offset = 0.6
						triangle(a,b*(base**(0*offset)),order+1,scale,offsets,ax)
						triangle(a,b*(base**(scale*(1*order + 1) + 1*offset)),order,scale,offsets,ax)
						triangle(a,b*(base**(scale*(2*order + 1) + 2*offset)),order-1,scale,offsets,ax)

	file = 'scaling_p%d_n%d_k%d_K%d___NAME_____TYPE__.pdf'%(max(P),max(N),order,k[0])
	path = os.path.join(directory,file)

	if 'gamma' in names:
		props = {
			'subplots_adjust':{'wspace':0.5,'hspace':0.5},
			'tight_layout':{'pad':300,'h_pad':100},								   						
			'savefig':{'fname':path,'bbox_inches':'tight'},
			}
	else:
		props = {
			'subplots_adjust':{'wspace':1,'hspace':1},
			# 'tight_layout':{'pad':500,'h_pad':100},								   						
			'savefig':{'fname':path,'bbox_inches':'tight'},
			}			

	for prop in props:
		for typed in types:
			for name in names:
				fig = figs[typed][name]
				if prop in ['savefig']:
					props[prop]['fname'] = props[prop]['fname'].replace("__NAME__",name).replace("__TYPE__",typed)
				getattr(fig,prop)(**props[prop])
				if prop in ['savefig']:
					props[prop]['fname'] = props[prop]['fname'].replace(name,"__NAME__").replace(typed,"__TYPE__")
	return

if __name__ == '__main__':
	defaults = ['analysis__*.pickle']
	paths = sys.argv[1:]
	if len(paths) == 0:
		paths.extend(defaults)
	scaling(paths)
