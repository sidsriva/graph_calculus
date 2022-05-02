#!/usr/bin/env python

# Import python modules
import os,sys,copy,warnings,functools,itertools
from natsort import natsorted
import numpy as np
import scipy as sp
import scipy.stats,scipy.signal
import pandas as pd
#from numba import jit

from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV,KFold,RepeatedKFold,TimeSeriesSplit,ShuffleSplit, GroupShuffleSplit
from sklearn import linear_model

# import multiprocess as mp
# import multithreading as mt
import multiprocessing as mp

warnings.simplefilter("ignore", (UserWarning,DeprecationWarning,FutureWarning))

# # Global Variables
DELIMITER = '__'

# Import user modules		
from utility.load_dump import load,dump,path_split,path_join
from utility.graph_utilities import set_loss,set_score,set_norm,set_scale,set_criteria
from utility.graph_utilities import reshaper,slice_index,ncombinations,isclose,isiterable
from utility.graph_utilities import polynomials,neighbourhood,vandermonde
from utility.graph_utilities import where,arange,invert,broadcast


# Logging
import logging
log = 'info'
logger = logging.getLogger(__name__)
#logger.setLevel(getattr(logging,log.upper()))






# Schemes for which data to use in fitting
def schemes(defaults={}):
	
	def default(X,y,shape,parameters,*args,**kwargs):
		inds = np.arange(shape[0])
		return X,y,[inds]*shape[2]

	def euler(X,y,shape,parameters,*args,**kwargs):
		default = 'fwd_diff'
		params = {'fwd_diff':{'weights':[0.0,1.0]}, 'bwd_diff':{'weights':[1.0,0.0]}, 'ctl_diff':{'weights':[0.5,0.5]}}
		params = params.get(parameters['method'],params[default])
		inds = np.arange(shape[0]-1)
		return np.concatenate(((params['weights'][1])*X[1:] + (params['weights'])*X[:-1],),axis=0),y[inds],[inds]*shape[2]

	def _wrapper(func):
		def _func(X,y,parameters,*args,**kwargs):
			parameters.update({k: defaults[k] for k in defaults if k not in parameters})
			ndim = X.ndim
			shape = list(X.shape[:3])
			if ndim < 3:
				shape.append(1)
			return func(X,y,shape,parameters,*args,**kwargs)
		return _func

	_defaults = {
		'scheme':None,
		'method':None,
		'iloc':None,
	}
	

	defaults.update({k:_defaults[k] for k in _defaults if k not in defaults})
	locs = locals()

	_schemes = {}
	_schemes.update({k: _wrapper(locs[k]) for k in locs if callable(locs[k]) and not k.startswith('_')})
	_schemes.update({None: _wrapper(default)})

	return _schemes




# Dataset features
def features(defaults={}):

	def default(X,y,rhs,lhs,parameters,*args,**kwargs):
		return X,y,rhs,lhs

	def linear(X,y,rhs,lhs,parameters,*args,**kwargs):
		axis = 1
		shape = list(X.shape)
		shape[axis] = 1
		if parameters['intercept_']:
			X = np.concatenate((np.ones(shape),X),axis=axis)
			rhs = [['intercept_',*r] for r in rhs] if (
					all([isinstance(r,list) for r in rhs])) else (
					['intercept_',*rhs])
		return X,y,rhs,lhs
	

	def monomial(X,y,rhs,lhs,parameters,*args,**kwargs):

		prefix = 'monomial'

		axis = 1
		shape = list(X.shape)
		shape[axis] = -1
		P = np.arange(1.0,parameters['order']+1.0)
		X = broadcast(X)
		X = np.moveaxis(np.power(X,P),-1,axis+1).reshape(shape,order='F')
		shape[axis] = 1
		if parameters['intercept_']:
			X = np.concatenate((np.ones(shape),X),axis=axis)
			rhs = ([[DELIMITER.join([prefix,str(None),'intercept_']),*[DELIMITER.join([prefix,str(None),'%s_%d'%(x,i)]) for i in np.arange(1,parameters['order']+1) for x in r]] for r in rhs] 
						if all([isinstance(r,list) for r in rhs]) else 
						[DELIMITER.join([prefix,str(None),'intercept_']),*[DELIMITER.join([prefix,str(None),'%s_%d'%(r,i)]) for i in range(1,parameters['order']+1) for r in rhs]])
		else:
			rhs = ([[DELIMITER.join([prefix,str(None),'%s_%d'%(x,i)]) for i in range(1,parameters['order']+1) for x in r] for r in rhs] 
						if all([isinstance(r,list) for r in rhs]) else 
						[DELIMITER.join([prefix,str(None),'%s_%d'%(x,i)]) for i in range(1,parameters['order']+1) for r in rhs])

		return X,y,rhs,lhs

	def polynomial(X,y,rhs,lhs,parameters,*args,**kwargs):
		
		prefix = 'polynomial'

		isxdim = X.ndim >= 3
		isydim = y.ndim >= 2
		if not isxdim:
			X = broadcast(X)
			rhs = [rhs]
		if not isydim == 1:
			y = broadcast(y)
			lhs = [lhs]


		n,p,d = X.shape
		_X = [None for dim in range(d)]
		_y = [None for dim in range(d)]
		_rhs = [None for dim in range(d)]
		_lhs = [None for dim in range(d)]
		for dim in range(d):
			_X[dim],_rhs[dim] = polynomials(
									X=X[...,dim],
									order=parameters['order'],
									derivative=0,
									selection=parameters['selection'],
									intercept_=parameters['intercept_'],
									variables=rhs[dim],
									return_poly=True,return_derivative=False,
									return_coef=False,return_polylabel=True,
									return_derivativelabel=False)




			_rhs[dim] = [DELIMITER.join([prefix,str(None),_rhs[dim][r]]) for r in _rhs[dim]]


			_y[dim] = y[...,dim]
			_lhs[dim] = lhs[dim]

		if isxdim:
			X = np.concatenate([broadcast(u) for u in _X],axis=-1)
			rhs = _rhs
		else:
			X = _X[0]
			rhs = _rhs[0]

		if isydim:
			y = np.concatenate([broadcast(u) for u in _y],axis=-1)
			lhs = _lhs
			pass
		else:
			y = np.concatenate([broadcast(u) for u in _y],axis=-1)
			lhs = lhs[0]

		return X,y,rhs,lhs

	

	def chebyshev(X,y,rhs,lhs,parameters,*args,**kwargs):

		prefix = 'chebyshev'

		axis = 1
		shape = list(X.shape)
		n,p = shape[0],shape[axis]    
		shape[axis] = -1
		X = np.polynomial.chebyshev.chebvander(X, parameters['order'])
		X = np.moveaxis(X,-1,axis+1).reshape(shape,order='F')
		X = X[:,p:]
		shape[axis] = 1	
		if parameters['intercept_']:
			X = np.concatenate((np.ones(shape),X),axis=axis)
			rhs = ([[DELIMITER.join([prefix,str(None),'intercept_']),*[DELIMITER.join([prefix,str(None),'%s_%d'%(x,i)]) for i in np.arange(1,parameters['order']+1) for x in r]] for r in rhs] 
						if all([isinstance(r,list) for r in rhs]) else 
						[DELIMITER.join([prefix,str(None),'intercept_']),*[DELIMITER.join([prefix,str(None),'%s_%d'%(r,i)]) for i in range(1,parameters['order']+1) for r in rhs]])
		else:
			rhs = ([[DELIMITER.join([prefix,str(None),'%s_%d'%(x,i)]) for i in range(1,parameters['order']+1) for x in r] for r in rhs] 
						if all([isinstance(r,list) for r in rhs]) else 
						[DELIMITER.join([prefix,str(None),'%s_%d'%(r,i)]) for i in range(1,parameters['order']+1) for r in rhs])

		return X,y,rhs,lhs


	def legendre(X,y,rhs,lhs,parameters,*args,**kwargs):

		prefix = 'legendre'

		axis = 1
		shape = list(X.shape)
		n,p = shape[0],shape[axis]    
		shape[axis] = -1
		X = np.polynomial.legendre.legvander(X, parameters['order'])
		X = np.moveaxis(X,-1,axis+1).reshape(shape,order='F')
		X = X[:,p:]		
		shape[axis] = 1
		if parameters['intercept_']:
			X = np.concatenate((np.ones(shape),X),axis=axis)
			rhs = ([[DELIMITER.join([prefix,str(None),'intercept_']),*[DELIMITER.join([prefix,str(None),'%s_%d'%(x,i)]) for i in np.arange(1,parameters['order']+1) for x in r]] for r in rhs] 
						if all([isinstance(r,list) for r in rhs]) else 
						[DELIMITER.join([prefix,str(None),'intercept_']),*[DELIMITER.join([prefix,str(None),'%s_%d'%(r,i)]) for i in range(1,parameters['order']+1) for r in rhs]])
		else:
			rhs = ([[DELIMITER.join([prefix,str(None),'%s_%d'%(x,i)]) for i in range(1,parameters['order']+1) for x in r] for r in rhs] 
						if all([isinstance(r,list) for r in rhs]) else 
						[DELIMITER.join([prefix,str(None),'%s_%d'%(r,i)]) for i in range(1,parameters['order']+1) for r in rhs])

		return X,y,rhs,lhs

	def hermite(X,y,rhs,lhs,parameters,*args,**kwargs):

		prefix = 'hermite'

		axis = 1
		shape = list(X.shape)
		n,p = shape[0],shape[axis]    
		shape[axis] = -1
		X = np.polynomial.hermite_e.hermevander(X, parameters['order'])
		X = np.moveaxis(X,-1,axis+1).reshape(shape,order='F')
		X = X[:,p:]		
		shape[axis] = 1
		if parameters['intercept_']:
			X = np.concatenate((np.ones(shape),X),axis=axis)
			rhs = ([[DELIMITER.join([prefix,str(None),'intercept_']),*[DELIMITER.join([prefix,str(None),'%s_%d'%(x,i)]) for i in np.arange(1,parameters['order']+1) for x in r]] for r in rhs] 
						if all([isinstance(r,list) for r in rhs]) else 
						[DELIMITER.join([prefix,str(None),'intercept_']),*[DELIMITER.join([prefix,str(None),'%s_%d'%(r,i)]) for i in range(1,parameters['order']+1) for r in rhs]])
		else:
			rhs = ([[DELIMITER.join([prefix,str(None),'%s_%d'%(x,i)]) for i in range(1,parameters['order']+1) for x in r] for r in rhs] 
						if all([isinstance(r,list) for r in rhs]) else 
						[DELIMITER.join([prefix,str(None),'%s_%d'%(r,i)]) for i in range(1,parameters['order']+1) for r in rhs])

		return X,y,rhs,lhs				



	def _wrapper(func):
		def _func(X,y,rhs,lhs,parameters,*args,**kwargs):
			parameters.update({k: defaults[k] for k in defaults if k not in parameters})
			return func(X,y,rhs,lhs,parameters,*args,**kwargs)
		return _func

	_defaults = {
		'order':1,
		'intercept_':False,
		'selection':None,
	}
	defaults.update({k:_defaults[k] for k in _defaults if k not in defaults})
	locs = locals()

	_features = {}
	_features.update({k: _wrapper(locs[k]) for k in locs if callable(locs[k]) and not k.startswith('_')})
	_features.update({None: _wrapper(default)})

	return _features


# Models
def models(defaults={}):

	# Expand series term with variables
	def _expansion(df,df0,label,variable,order,variables,iloc=None,groupby=False,unique=True):
		try:
			if iloc is not None:
				if groupby:
					df = df.transform('nth',iloc)
				else:
					df = df.iloc[iloc]
			try:
				function = (df0[label].values)
			except:
				try:
					function = df0[label]
				except:
					function = 1
			if order > 0:
				difference = (((df[variable].values-df0[variable].values).prod(axis=1)))
			else:
				difference = 1
			try:
				pass
			except:
				pass
			if unique and order > 0:
				factor = sp.special.factorial([variable.count(v) for v in variables],exact=True).prod()
			else:
				factor = sp.special.factorial(order,exact=True)
			out = ((function)*(difference))/factor + 0.0
			return out
		except Exception as e:
			raise e
		return



	def taylorseries(df,df0,rhs,lhs,parameters,*args,**kwargs):

		# Get testing data df and fitting data df0 size
		n,m = df.shape[0],df0.shape[0]

		# Get number of dimensions in model
		d = min(len(rhs),len(lhs))

		# Get number of dimensions in model manifold, order of model, terms in model, and operators included in taylor series
		p = parameters['p']
		order = parameters['order']
		terms = parameters['terms']
		operations = ['partial']
		
		# Get iloc locations of data in reference df0 to fit model. 
		# If None, indicates that for each point in df, a nearest point of the models in df0 must be found,
		# If list[int], is list of points in df0 of length d, with parallel list in rhs,lhs for terms in model for each point
		
		# Get ilocs location(s) of allowed points in df0 to base models
		# If iloc is None, indicates that fitting different models at all points in df0 and rhs,lhsare  expanded to be length d = m different models

		# iloc is None or list[int] of length d of allowed locations in df0 to base model with df data, elements of iloc MUST be in ilocs
		# ilocs is list[int] of allowed locations in df0 to expand model for fitting with data in df 
		iloc = parameters['iloc']
		ilocs = parameters['ilocs']

		if ilocs in [None]:
			ilocs = list(range(m))

		# Get whether there are multiple models to be fit
		multiple = len(ilocs) > 1

		# Get parameter around which to base taylor series (forms different taylor series at each unique parameter value in manifold)
		parameter = parameters['parameter']

		# Basis of model
		basis = vandermonde

		# Get parameters for neighbourhood finding, tolerances between points, sorting algorithm,
		# whether taylor series terms include all unique commuting derivatives, or non-commuting non-unique derivatives,
		# chunk size of adjacency calculations,		
		# and whether to find strict size of neighbourhood
		tol = parameters['tol']
		atol = parameters['atol']
		rtol = parameters['rtol']
		kind = 'mergesort'
		unique = parameters['unique']
		chunk = parameters['chunk']		
		strict = False
		metric = parameters['metric']
		format = parameters['format']
		dtype = parameters['dtype']
		prefix = 'taylorseries'	

		# Get size of neighbourhood of points in df0 nearest to points in df to consider based on order of taylor series
		if order is None:
			size = min(n,m)
		else:
			size = ncombinations(p,order,unique=unique)

		# Specify rhs,lhs if iloc for df is None
		# Get iloc (which will be list of length d)

		# If iloc is None:
		# lhs is list of unique lhs models in lhs of length d
		# rhs is list of all terms that satisfy constraints for each lhs model
		# iloc is [None for i in range(d)]

		# Otherwise rhs,lhs,iloc are not updated and are lists of length d
		if iloc is None:
			lhs = list(sorted(set(lhs),key=lambda x:lhs.index(x)))
			rhs = [[term['label'][-1] for term in terms
					if ((term['function'] == l) and 
						(all([o in operations for o in term['operation']])))
					]
					for l in lhs]
			iloc = [iloc for r,l in zip(rhs,lhs)]
		else:
			lhs = [l for l in lhs]
			rhs = [r for r in rhs]
			iloc = iloc

		# Get number of dimensions, and number of variables in rhs,lhs
		d,q = np.shape(rhs)

		# Get data X,y		
		X = np.zeros((n,q,d))
		y = np.zeros((n,d))


		# Get list of allowed dims
		dims = list(range(d))


		# Get fields of terms for variables 
		fields = {'label':-1,'variable':None,'order':None,'manifold':-1}

		# Get variables fields, distances, indices, and neighbourhoods of points in df closest to df0 for fitting points nearest to each model in df0
		variables = [{field:[] for field in fields} for dim in dims]		
		var = [[] for dim in dims]
		distances = [[0] for dim in dims]
		indices = [[0] for dim in dims]
		neighbourhoods = [None for dim in dims]
		neighbourhoodsT = [None for dim in dims]
		_df0 = [None for dim in dims]


		# Get whether each iloc is an integer location in df0, or None to be found
		integer = [isinstance(iloc[dim],(int,np.integer)) for dim in dims]

		# For each dimension of rhs,lhs, get variables
		for dim in dims:
			r = rhs[dim]
			l = lhs[dim]

			# Get variables (currently the same variables for each dimension)
			for term in terms:
				if all([o in operations for o in term['operation']]) and (l==term['function']):					
					for field in fields:
						index = fields[field]
						try:
							variables[dim][field].append(term[field][index])
						except:
							variables[dim][field].append(term[field])

			field = 'manifold'
			var[dim] = list(list(set((tuple(v) for v in variables[dim][field] if len(v)>0)))[0])


		# Get neighbourhoods of indices of nearest points in df0 to df with variables data, shape (d,size) (using euclidean 2-norm between data)
		# result = (indices,distances) of nearest points for each unique variable set

		# Unique sets of variables to search manifold of df-df0, 
		# and indices of each var that equals each unique varset
		varset = [list(v) for v in list(set((tuple(v) for v in var)))]
		varindices = [[dim for dim in dims if var[dim] == v] for v in varset]
		nvar = len(varset)


		for i in range(nvar):


			# Get indices of size nearest points of df to each of df0 of shape (m,size) (subject to linearly independence constraints on points for model basis)
			# indices are in {0,1,2,....,n-1}, for all indices in df
			result = neighbourhood(df0,df,size=size,basis=basis,
						variable=varset[i],
						indices=None,order=order,metric=metric,unique=unique,diagonal=True,chunk=chunk,
						argsortbounds=[None,None,None],											
						weightbounds=[None,None,None],
						tol=tol,atol=atol,rtol=rtol,
						format=format,dtype=dtype,
						strict=strict,kind=kind,
						return_weights=True,
						)
			
			# Get indices of nearest points of df0 (within allowed models at ilocs) to each of df of shape (n,1)
			# indices are in indices of ilocs, for all allowed model locations in df0
			resultT = neighbourhood(df,df0,size=1,basis=basis,
						variable=varset[i],
						indices=ilocs,order=order,metric=metric,unique=unique,diagonal=True,chunk=chunk,
						argsortbounds=[None,None,None],											
						weightbounds=[None,None,None],
						tol=tol,atol=atol,rtol=rtol,
						format=format,dtype=dtype,						
						strict=strict,kind=kind,
						return_weights=True,
						)	

			for dim in varindices[i]:		
				neighbourhoods[dim] = result
				neighbourhoodsT[dim] = resultT


		# For each dimension of rhs,lhs, get indices and iloc for rhs expansion based on neighbourhood
		for dim in dims:
			r = rhs[dim]
			l = lhs[dim]			


			# Get iloc and distance of data in df0 nearest to fit data in df

			# If iloc is already integer, do not update and is int
			# If iloc is None, use nearest locations found above for each point in df and is array of length n
			if integer[dim]:
				iloc[dim] = iloc[dim]
				distances[dim] = np.abs(neighbourhoods[dim][1][iloc[dim]])
			else:
				iloc[dim] = [i for i in neighbourhoodsT[dim][0]]
				distances[dim] = np.abs(neighbourhoodsT[dim][1][0])


			# Get indices of fitting

			# If iloc is specific integer (fitting at certain point in df0), there are multiple ilocs models to consider, and df0==df, 
			# then indices of fitting are neighbourhood of k indices from above for fitting model
			# If iloc is None, or ilocs is single location, or df0 != df
			# then indices of fitting are all indices of df

			use = integer[dim] and multiple and isclose(df0[var[dim]].values,df[var[dim]].values).all()

			if use:
				indices[dim] = neighbourhoods[dim][0][iloc[dim]]
			else:
				indices[dim] = slice(n)



			# Choose location in df0, iloc, to base model
			if parameter in df0 or (isiterable(parameter) and all([param in df0 for param in parameter])):
				_df0 = df0.groupby(by=parameter,as_index=False).transform('nth',iloc[dim])
			else:
				_df0 = df0.iloc[iloc[dim]]


			# Update rhs labels with expansion prefix and iloc (int value if integer else None)
			rhs[dim] = [DELIMITER.join([prefix if any([u.startswith(o) for o in operations]) else 'constant',str(iloc[dim]) if integer[dim] else str(None),str(u)]) for u in r]


			for index,(l,v,o,V) in enumerate(zip(variables[dim]['label'],
											 variables[dim]['variable'],
											 variables[dim]['order'],
											 variables[dim]['manifold'])):				
				values = _expansion(df=df,df0=_df0,label=l,variable=v,order=o,variables=V,unique=unique)

				X[:,index,dim] = values

			y[:,dim] = df[lhs[dim]].values



		# Get unique iloc locations for each dimension, as lists of unique possible iloc locations for that dimension model

		# Make ilocations a list of length d, with each element a dictionary of 
		# keys of all ilocs used to fit df with respect to ilocs in df0 for that dimension
		# values of all locations in df that will be fit based on the model at ilocs key
		ilocations = [np.array(iloc[dim]).reshape(-1) for dim in dims]
		ilocations = [{i: arange(n)[where(ilocations[dim]==i)] for i in ilocs if i in ilocations[dim]} for dim in dims]

		# Update parameters
		parameters['dims'] = dims
		parameters['iloc'] = iloc
		parameters['ilocs'] = ilocs
		parameters['ilocations'] = ilocations
		parameters['distances'] = distances
		parameters['indices'] = indices

		# Use only data at indices for model
		X = np.concatenate([X[indices[dim]][...,dim][...,None] for dim in dims],axis=-1)
		y = np.concatenate([y[indices[dim]][...,dim][...,None] for dim in dims],axis=-1)
		return X,y,rhs,lhs


	def default(df,df0,rhs,lhs,parameters,*args,**kwargs):
		X = np.concatenate(tuple([df[r].values[...,None] for r in rhs]),axis=-1)
		y = df[lhs].values
		X,y,rhs,lhs = _features['default'](X,y,rhs,lhs,parameters,**kwargs)
		return X,y,rhs,lhs

	def polynomial(df,df0,rhs,lhs,parameters,*args,**kwargs):
		X = np.concatenate(tuple([df[r].values[...,None] for r in rhs]),axis=-1)
		y = df[lhs].values
		X,y,rhs,lhs = _features['polynomial'](X,y,rhs,lhs,parameters,**kwargs)
		return X,y,rhs,lhs

	def monomial(df,df0,rhs,lhs,parameters,*args,**kwargs):
		X = np.concatenate(tuple([df[r].values[...,None] for r in rhs]),axis=-1)
		y = df[lhs].values
		X,y,rhs,lhs = _features['monomial'](X,y,rhs,lhs,parameters,**kwargs)
		return X,y,rhs,lhs

	def linear(df,df0,rhs,lhs,parameters,*args,**kwargs):
		X = np.concatenate(tuple([df[r].values[...,None] for r in rhs]),axis=-1)
		y = df[lhs].values
		X,y,rhs,lhs = _features['linear'](X,y,rhs,lhs,parameters,**kwargs)
		return X,y,rhs,lhs

	def chebyshev(df,df0,rhs,lhs,parameters,*args,**kwargs):
		X = np.concatenate(tuple([df[r].values[...,None] for r in rhs]),axis=-1)
		y = df[lhs].values
		X,y,rhs,lhs = _features['chebyshev'](X,y,rhs,lhs,parameters,**kwargs)
		return X,y,rhs,lhs

	def legendre(df,df0,rhs,lhs,parameters,*args,**kwargs):
		X = np.concatenate(tuple([df[r].values[...,None] for r in rhs]),axis=-1)
		y = df[lhs].values	
		X,y,rhs,lhs = _features['legendre'](X,y,rhs,lhs,parameters,**kwargs)
		return X,y,rhs,lhs	

	def hermite(df,df0,rhs,lhs,parameters,*args,**kwargs):
		X = np.concatenate(tuple([df[r].values[...,None] for r in rhs]),axis=-1)
		y = df[lhs].values
		X,y,rhs,lhs = _features['hermite'](X,y,rhs,lhs,parameters,**kwargs)
		return X,y,rhs,lhs								





	def _wrapper(func,name):

		@functools.wraps(func)
		def _func(df,df0,rhs,lhs,parameters,*args,**kwargs):
			parameters.update({k: defaults[k] for k in defaults if k not in parameters})
			
			expand = parameters['expand']


			if name not in ['taylorseries']:
			
				# Get testing data df and fitting data df0 size
				n,m = df.shape[0],df0.shape[0]

				# Get number of dimensions in model
				d = min(len(rhs),len(lhs))

				# Get number of dimensions in model manifold, order of model, terms in model, and operators included in model
				p = parameters['p']
				order = parameters['order']
				terms = parameters['terms']
				operations = ['partial']
				
				# Get iloc locations of data in reference df0 to fit model. 
				# If None, indicates that for each point in df, a nearest point of the models in df0 must be found,
				# If list[int], is list of points in df0 of length d, with parallel list in rhs,lhs for terms in model for each point
				
				# Get ilocs location(s) of allowed points in df0 to base models
				# If iloc is None, indicates that fitting different models at all points in df0 and rhs,lhsare  expanded to be length d = m different models

				# iloc is None or list[int] of length d of allowed locations in df0 to base model with df data, elements of iloc MUST be in ilocs
				# ilocs is list[int] of allowed locations in df0 to expand model for fitting with data in df 
				iloc = parameters['iloc']
				ilocs = parameters['ilocs']

				if iloc is None:
					iloc = [iloc for r,l in zip(rhs,lhs)]

				if ilocs in [None]:
					ilocs = list(range(m))


				# Get whether there are multiple models to be fit
				multiple = len(ilocs) > 1

				# Get parameter around which to base model (forms different model at each unique parameter value in manifold)
				parameter = parameters['parameter']

				# Get basis of model
				basis = vandermonde

				# Get parameters for neighbourhood finding, tolerances between points, sorting algorithm,
				# whether model terms include all unique commuting derivatives, or non-commuting non-unique derivatives,
				# chunk size of adjacency calculations,
				# and whether to find strict size of neighbourhood
				tol = parameters['tol']
				atol = parameters['atol']
				rtol = parameters['rtol']
				kind = 'mergesort'
				unique = parameters['unique']
				chunk = parameters['chunk']
				strict = False
				metric = parameters['metric']
				format = parameters['format']
				dtype = parameters['dtype']
				manifold = parameters['manifold']

				# Get number of dimensions, and number of variables in rhs,lhs
				d,q = np.shape(rhs)


				# Get size of neighbourhood of points in df0 nearest to points in df to consider based on order of model
				if order is None:
					size = m
				else:
					size = ncombinations(p,order,unique=unique) if name in ['polynomial'] else order+1


				# Get data X,y		
				X = np.zeros((d,n,q))
				y = np.zeros((d,n))


				# Get list of allowed dims
				dims = list(range(d))

				# Get fields of terms for variables 
				fields = {'label':-1,'variable':None,'order':None,'variables':-1}

				# Get variables fields, distances, indices, and neighbourhoods of points in df closest to df0 for fitting points nearest to each model in df0
				variables = [{field:[] for field in fields} for dim in dims]		
				var = [[] for dim in dims]
				distances = [[0] for dim in dims]
				indices = [[0] for dim in dims]
				neighbourhoods = [None for dim in dims]
				neighbourhoodsT = [None for dim in dims]
				_df0 = [None for dim in dims]


				# Get whether each iloc is an integer location in df0, or None to be found
				integer = [isinstance(iloc[dim],(int,np.integer)) for dim in dims]

				# For each dimension of rhs,lhs, get variables
				for dim in dims:
					r = rhs[dim]
					l = lhs[dim]

					# Get variables (currently the same variables for each dimension)
					var[dim] = manifold if manifold is not None else rhs[dim]



				# Get neighbourhoods of indices of nearest points in df0 to df with variables data, shape (d,size) (using euclidean 2-norm between data)
				# result = (indices,distances) of nearest points for each unique variable set

				# Unique sets of variables to search manifold of df-df0, 
				# and indices of each var that equals each unique varset
				varset = [list(v) for v in list(set((tuple(v) for v in var)))]
				varindices = [[dim for dim in dims if var[dim] == v] for v in varset]
				nvar = len(varset)

				for i in range(nvar):


					# Get indices of size nearest points of df to each of df0 of shape (m,size) (subject to linearly independence constraints on points for model basis)
					# indices are in {0,1,2,....,n-1}, for all indices in df
					result = neighbourhood(df0,df,size=size,basis=basis,
								variable=varset[i],
								indices=None,order=order,metric=metric,unique=unique,diagonal=True,chunk=chunk,
								argsortbounds=[None,None,None],											
								weightbounds=[None,None,None],
								tol=tol,atol=atol,rtol=rtol,
								format=format,dtype=dtype,
								strict=strict,kind=kind,
								return_weights=True,
								)
					
					# Get indices of nearest points of df0 (within allowed models at ilocs) to each of df of shape (n,1)
					# indices are in indices of ilocs, for all allowed model locations in df0
					resultT = neighbourhood(df,df0,size=1,basis=basis,
								variable=varset[i],
								indices=ilocs,order=order,metric=metric,unique=unique,diagonal=True,chunk=chunk,
								argsortbounds=[None,None,None],											
								weightbounds=[None,None,None],
								tol=tol,atol=atol,rtol=rtol,
								format=format,dtype=dtype,								
								strict=strict,kind=kind,
								return_weights=True,
								)	

					for dim in varindices[i]:		
						neighbourhoods[dim] = result
						neighbourhoodsT[dim] = resultT



				# For each dimension of rhs,lhs, get indices and iloc for rhs expansion based on neighbourhood
				for dim in dims:
					r = rhs[dim]
					l = lhs[dim]			


					# Get iloc and distance of data in df0 nearest to fit data in df

					# If iloc is already integer, do not update and is int
					# If iloc is None, use nearest locations found above for each point in df and is array of length n
					if integer[dim]:
						iloc[dim] = iloc[dim]
						distances[dim] = np.abs(neighbourhoods[dim][1][iloc[dim]])
					else:
						iloc[dim] = [i for i in neighbourhoodsT[dim][0]]
						distances[dim] = np.abs(neighbourhoodsT[dim][1][0])


					# Get indices of fitting

					# If iloc is specific integer (fitting at certain point in df0), there are multiple ilocs models to consider, and df0==df, 
					# then indices of fitting are neighbourhood of k indices from above for fitting model
					# If iloc is None, or ilocs is single location, or df0 != df
					# then indices of fitting are all indices of df

					use = integer[dim] and multiple and isclose(df0[var[dim]].values,df[var[dim]].values).all()

					if use:
						indices[dim] = neighbourhoods[dim][0][iloc[dim]]
					else:
						indices[dim] = slice(n)

					# Choose location in df0, iloc, to base model
					if parameter in df0 or (isiterable(parameter) and all([param in df0 for param in parameter])):
						_df0[dim] = df0.groupby(by=parameter,as_index=False).transform('nth',iloc[dim])
					else:
						_df0[dim] = df0.iloc[iloc[dim]]


					if expand:
						X[dim] = df[r].values - _df0[dim][r].values		
					else:
						X[dim] = df[r].values

					y[dim] = df[lhs[dim]].values


				X = np.concatenate(tuple([x[...,None] for x in X]),axis=-1)
				y = np.concatenate(tuple([u[...,None] for u in y]),axis=-1)


				# Get model basis
				X,y,rhs,lhs = _features[name](X,y,rhs,lhs,parameters,**kwargs)


				if expand:
					X = [x for x in X.transpose((2,0,1))]

					for dim in dims:
						r = rhs[dim]
						l = lhs[dim]						
						constants = _df0[dim][l]
						if constants.size == 1: 
							constants = np.ones((X[dim].shape[0],1))*_df0[dim][l]
						else:
							constants = constants[...,None]
						prefix = 'polynomial'
						constant = DELIMITER.join([prefix,str(None),l])

						X[dim] = np.concatenate((constants,X[dim]),axis=-1)
						r.insert(0,constant)


					prefix = ''
					X = np.concatenate(tuple([x[...,None] for x in X]),axis=-1)
					rhs = [[u.replace(DELIMITER.join([prefix,str(None),'']),DELIMITER.join([prefix,str(iloc[dim]) if integer[dim] else str(None),''])) for u in rhs[dim]] for dim in dims]





				# Get unique iloc locations for each dimension, as lists of unique possible iloc locations for that dimension model

				# Make ilocations a list of length d, with each element a dictionary of 
				# keys of all ilocs used to fit df with respect to ilocs in df0 for that dimension
				# values of all locations in df that will be fit based on the model at ilocs key
				ilocations = [np.array(iloc[dim]).reshape(-1) for dim in dims]
				ilocations = [{i: arange(n)[where(ilocations[dim]==i)] if not integer[dim] else arange(n) for i in ilocs if i in ilocations[dim]} for dim in dims]

				# Update parameters
				parameters['dims'] = dims
				parameters['iloc'] = iloc
				parameters['ilocs'] = ilocs
				parameters['ilocations'] = ilocations
				parameters['distances'] = distances
				parameters['indices'] = indices

				# Use only data at indices for model
				X = np.concatenate([X[indices[dim]][...,dim][...,None] for dim in dims],axis=-1)
				y = np.concatenate([y[indices[dim]][...,dim][...,None] for dim in dims],axis=-1)

			else:
				
				X,y,rhs,lhs = func(df,df0,rhs,lhs,parameters,*args,**kwargs)
			
			return X,y,rhs,lhs

		return _func

	_defaults = {
		'iloc':None,
		'terms':[],
		'parameter':None,
		'intercept_':False,
		'expand':False,
	}
	defaults.update({k:_defaults[k] for k in _defaults if k not in defaults})
	locs = locals()


	_features = {}
	_features.update(features())
	_models = {}
	_models.update({k: _wrapper(locs[k],k) for k in locs if callable(locs[k]) and not k.startswith('_')})
	_models.update({None: _wrapper(default,'default')})

	return _models

def model(data,metadata,settings,funcs,verbose=False,**kwargs):

	def _model(df0,parameters={},**kwargs):

		def wrapper(df,rhs,lhs,params={}):
		
			# Update params with parameters, making plural key with 's' if params already has parameter key
			for k in parameters:
				exists = k in params
				v = copy.deepcopy(parameters[k])
				_k = '%s%s'%(k,'' if not exists else 's')
				_v = v if not exists else v #([v for i in range(len(v))] if isinstance(v,list) else v)
				params[_k] = _v
			
			locs = locals()
			fields = {k: locs.get(k) for k in ['kwargs']}
			for field in fields:
				value = fields[field]
				try:
					params[field].update(value)
				except:
					params[field] = value


			_models = models()[params['basis']]
			_features = features()[params['transform']['features']['features']]
			_schemes = schemes()[params['transform']['scheme']['scheme']]
			_scale = set_scale(**params['transform']['normalization'])

			# Get rhs,lhs
			if not all([isinstance(r,list) for r in rhs]):
				rhs = [rhs]
			if not isinstance(lhs,list):
				lhs = [lhs]


			n = df.shape[0]

			X,y,rhs,lhs = _models(df,df0,rhs,lhs,params,**kwargs)
			X,y,rhs,lhs = _features(X,y,rhs,lhs,params['transform']['features'],**kwargs)

			X,y,inds = _schemes(X,y,params['transform']['scheme'],**kwargs)

			if y.ndim < 2:
				y = broadcast(y)
			if X.ndim < 3:
				X = broadcast(X)

			X = X.astype(params['transform']['dtype'])
			y = y.astype(params['transform']['dtype'])

			if params.get('scale') is None:
				params['scale'] = {}
			if params.get('indices') is None:
				params['indices'] = inds
			else:
				params['indices'] = [slice_index(i,j,n) for i,j in zip(params['indices'],inds)]


			params['scale']['X'] = _scale(X)
			params['scale']['y'] = _scale(y)

			return X,y,rhs,lhs


		# Set default parameters
		defaults = {
			'basis':None,
			'order':None,
			'iloc':[0],
			'p':None,
			'parameter':None,
			'selection':None,
			'intercept_':False,
			'transform':{
				'features':{
					'order':None,'features':None,
					'p':None,'intercept_':False,'selection':None,
					'iterate':False},
				'normalization':{
					'axis':0,'normalization':'l2'},
				'scheme':{
					'scheme':None,'method':None,'adjacency':None},
				'dtype':'float64',
				}
			}

		for parameter in defaults:
			if parameter not in parameters:
				parameters[parameter] = copy.deepcopy(defaults[parameter])
			elif isinstance(parameters[parameter],dict):
				parameters[parameter].update({u:defaults[parameter][u] for u in defaults.get(parameter,[]) if u not in parameters[parameter]})


		return wrapper



	# Setup fields  and rhs_lhs in metadata
	setup(data,metadata,settings,funcs,verbose=verbose)


	# Setup models
	for label in funcs:		
		for key in data:
			if key in funcs[label]:
				funcs[label][key] = _model(
										data[key],
										metadata[key],
										**kwargs)


	return



# Set default model settings
def setup(data,metadata,settings,funcs,verbose=False):
	fields = {
			'rhs_lhs':{},'basis':None,'n':None,'p':None,'transform':None,'scheme':None,
			'order':None,'selection':None,'intercept_':None,'terms':[],
			'iloc':0,'accuracy':None,'tol':None,'atol':None,'rtol':None,
			**settings['model'],
			}

	for key in data:

		# Update missing fields with settings, possible key dependent values
		for field in fields:
			value = copy.deepcopy(metadata[key].get(field,settings['model'].get(field,fields[field])))
			try:
				metadata[key][field] = value[key]
			except:
				metadata[key][field] = value

			if isinstance(metadata[key][field],dict):
				for parameter in metadata[key][field]:
					value = copy.deepcopy(metadata[key][field][parameter])
					try:
						metadata[key][field][parameter] = value[key]
					except:
						metadata[key][field][parameter] = value


			# Handle exceptional fields

			# Handle iloc field, depending if iloc is bool,int,float, or iterable and make into list

			# iloc can be passed as one of five types:
			# model fitting uses reference dataset df0 and given dataset df to form model in graph_fit and graph_models
			# 1) int : specific location in df0 to expand model and fit with data in df
			# 2) iterable[int]: several specific locations in df0 to expand model and fit with data in df
			# 3) float: proportion of locations in df0 to expand number, chooses randomly
			# 3) True: all allowed locations in ilocs of df0 
			# 4) None: Find closest point in ilocs of df0 for each point in df
			# Ensure iloc is either dictionary of {dataset:value} value, where value is 1) to 4) of above
			# If iloc type is an int,float,True then in graph_models, iloc parameter is pre-processed into list, depending on specific type and size of df0
			# If iloc is None, then kept as None and processed in graph_models when forming specific model/finding nearest points for fitting

			if field in ['iloc']:
				value = metadata[key][field]

				size = data[key].shape[0]

				if value in [None]:
					value = [None]
				if isinstance(value,(bool,int,np.integer,float,np.float)):
					value = [value]
				elif isinstance(value,np.ndarray):
					value = value.tolist()

				if len(value) == 1:
					value = value.pop(0)
					if value in [None]:
						value = value
					elif isinstance(value,bool) and value:
						value = list(range(size))
					elif isinstance(value,bool) and not value:
						value = [int(value)]
					elif isinstance(value,(int,np.integer)):
						value = [value]
					elif isinstance(value,(float,np.float)):
						value = np.random.choice(np.arange(size),int(value*size),replace=False).tolist()
						value = (i for i in value)
					else:
						value = [value]


				typed = list
				assert (value is None) or isinstance(value,typed),"Error - %s value is not None or of type %r"%(field,typed)

				metadata[key][field] = value

	# Setup rhs lhs of models
	rhs_lhs(data,metadata,settings,funcs,verbose=verbose)

	return

# Choose operators based on certain keywords
# Model could be for one validation data set [indiv] or multiple [multi] and then can either be functional or taylorseries
def rhs_lhs(data,metadata,settings,funcs,verbose=False):
	
	field = 'rhs_lhs'
	
	for key in data:

		# Ensure field is in metadata
		if metadata[key].get(field) is None:
			metadata[key][field] = {}



		# Get parameters for setting rhs,lhs based on terms in model, inputs and outputs
		terms = metadata[key]['terms']
		inputs = metadata[key]['inputs']
		outputs = metadata[key]['outputs']

		# iloc is list of allowed locations to base models at, handling None separately, but still ensuring at least one model
		ilocs = [i for i in metadata[key]['iloc']] if metadata[key].get('iloc') not in [None] else [metadata[key]['iloc']]

		# Order of basis
		order = metadata[key]['order']

		# Types of bases
		basises = {
				**{k:[None,'default','linear','monomial','polynomial','chebyshev','legendre','hermite'] for k in ['functional']},
				**{k: ['taylorseries'] for k in ['series']}
			}



		# RHS and LHS
		# rhs and lhs are lists of length l for l different models {model_i, i in range(l)}
		# rhs = [[[rhs_j_k for k in range(q_i_j)] for j in range(d_i)] for i range(l)] 
		# is list of list of lists, corresponding to list of q_i_j rhs terms (each a dataframe column label) for each dimension d_i length model_i
		# lhs = [[lhs_i of dimension d_i] for i range(l)]
		# is list of lists for each lhs term (a dataframe column label) for each dimension d_i length model_i

		# The length of models l is related to number of outputs
		# The dimension of each model d_i is related to number of iloc positions for model
		# The number of terms in each model dimension q_i_j is related to number of inputs

		# If approach is individual (indiv), l = len(outputs), and d_i = len(ilocs)
		# If approach is multiply (multi), l = 1 and d_i = len(outputs)*len(ilocs), and d_i
		# Number of terms in model is q_i_j = len(inputs) or len(terms) in case of Taylor series

		# rhs,lhs lists of length l are converted to dictionary of {label_i:{'rhs':rhs_i,'lhs_i'} for i in range(l)}
		# where as above:
		# rhs_i = [[rhs_j_k for k in range(q_i_j)] for j in range(d_i)]
		# lhs_i = [lhs_i of dimension d_i]

		if len(metadata[key][field])>0:
			

			for label in metadata[key][field]:
				rhs = metadata[key][field][label].get('rhs',[[]]) #rhs if list of lists of terms
				lhs = metadata[key][field][label].get('lhs',[])
				metadata[key][field][label] = {
						'rhs':[r if isinstance(r,list) else r for iloc in ilocs for r in rhs],
						'lhs':[l for iloc in ilocs for l in lhs],
						}

			# Update model funcs with label and keys
			if not isinstance(funcs.get(label),dict):
				funcs[label] = {}
			funcs[label].update({key:None})


			continue
		
		# Default
		sides = ['lhs','rhs']
		values = {}
		if terms is None:
			values['lhs'] = [[y for iloc in ilocs] for y in outputs]
			values['rhs'] = [[inputs for iloc in ilocs] for l in values['lhs']]			


		# Taylor Series
		elif settings['model']['approach'] in ['indiv'] and settings['model']['basis'] in basises['series'] and settings['model']['type'] in ['function']:
			values['lhs'] = [[y for iloc in ilocs ] for y in outputs ]
			values['rhs'] = [[[term['label'][-1] for term in terms if (
						(term['function'] == y) and (all([o in ['partial'] for o in term['operation']])))]
				   for iloc in ilocs
				   ]
				   for y in outputs
				   ]

		elif settings['model']['approach'] in ['multi'] and settings['model']['basis'] in basises['expansion'] and settings['model']['type'] in ['function']:
			values['lhs'] = [[y for y in outputs for iloc in ilocs]]						
			values['rhs'] = [[[term['label'][-1] for term in terms  if (
						(term['function'] == y) and (all([o in ['partial'] for o in term['operation']])))]
				   for y in outputs
				   for iloc in ilocs]
				   ]  



		for i,value in enumerate(zip(*[values[side] for side in sides])):

			# Label
			side = sides.index('lhs')
			label = DELIMITER.join([str(s) for s in [*sorted(set(value[side]),key=lambda x: value[side].index(x))]])

			# Assign rhs_lhs to metadata and 
			if not isinstance(metadata[key].get(field),dict):
				metadata[key][field] = {}
			if not isinstance(metadata[key][field].get(label),dict):
				metadata[key][field][label] = {}


			metadata[key][field][label] = dict(zip(sides,value))

			# Update model funcs with label and keys
			if not isinstance(funcs.get(label),dict):
				funcs[label] = {}
			funcs[label].update({key:None})





	return
