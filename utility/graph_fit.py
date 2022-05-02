#!/usr/bin/env python

# Import python modules
import os,sys,pickle,json,copy,warnings,itertools,timeit,datetime
import numpy as np
import pandas as pd
import scipy as sp
import scipy.stats,scipy.signal,scipy.cluster
warnings.simplefilter("ignore", (UserWarning,DeprecationWarning,FutureWarning))


# Global Variables
DELIMITER='__'

# Import user modules
from utility.estimator import Estimator,CrossValidate,Stepwise,OLS
from utility.dictionary import _get,_set
from utility.graph_utilities import invert,extrema,cond,rank,where,isin,boundaries,concatenate
from utility.graph_utilities import zeros,ones,arange,getsizeof



# Logging
import logging
log = 'info'
logger = logging.getLogger(__name__)
#logger.setLevel(getattr(logging,log.upper()))


# @profile
def fit(data,metadata,rhs,lhs,label,info,models,estimator,parameters,verbose=False):   


	def preprocess(info,label,parameters):


		X,y,rhs,lhs = info['model'](info['data']['data'],info['rhs'],info['lhs'],info)



		n,d = y.shape[:2]

		info['data']['data'] = info['data']['data']
		info['data']['X'] = X
		info['data']['y'] = y
		info['operators'] = {'X':rhs,'y':lhs}
		info['estimator'] = {}
		info['iloc'] = info['iloc']
		info['dims'] = info['dims']
		info['kwargs'] = {dim: {} for dim in info['dims']}


		if isinstance(info['groups'],np.ndarray):
			info['groups'] = [info['groups'] for i in info['indices']]


		for dim in info['dims']:
			info['kwargs'][dim]['loss_l2_inds'] = slice(None)
			info['kwargs'][dim]['loss_linf_inds'] = extrema(y[:,dim],**parameters.get('extrema',{})) if parameters.get('extrema') else slice(None)
			info['kwargs'][dim]['loss_l1_inds'] = extrema(y[:,dim],**parameters.get('extrema',{})) if parameters.get('extrema') else slice(None)
			info['kwargs'][dim]['loss_l2_scale'] = 1 #invert(y[:,dim],constant=1)
			info['kwargs'][dim]['loss_linf_scale'] = 1 #invert(y[:,dim],constant=1)
			info['kwargs'][dim]['loss_l1_scale'] = 1 #invert(y[:,dim],constant=1)
			info['kwargs'][dim]['score_l2_inds'] = slice(None)
			info['kwargs'][dim]['score_linf_inds'] = extrema(y[:,dim],**parameters.get('extrema',{})) if parameters.get('extrema') else slice(None)
			info['kwargs'][dim]['score_l1_inds'] = extrema(y[:,dim],**parameters.get('extrema',{})) if parameters.get('extrema') else slice(None)
			info['kwargs'][dim]['score_l2_scale'] = 1 #invert(y[:,dim],constant=1)
			info['kwargs'][dim]['score_linf_scale'] = 1 #invert(y[:,dim],constant=1)
			info['kwargs'][dim]['score_l1_scale'] = 1 #invert(y[:,dim],constant=1)			
			info['kwargs'][dim]['groups'] = info['groups'][dim]


			fields = ['loss_weights','score_weights','rcond','solver','included','fixed','cv']
			for field in fields:
				if field in parameters and field not in info['kwargs'][dim]:
					value = parameters[field]
					
					if field in ['loss_weights','score_weights']:
						assert sum([value[u] for u in value]) == 1.0, "Incorrect %s: %r"%(field,value)
					
					elif field in ['included']:
						func = lambda value,variable: np.array([variable.index(v) if (v in variable) else v for v in value
											if (v in variable) or ((isinstance(v,(int,np.integer)) and v<len(variable)))]).astype(int)
						variable = rhs[dim]
						value = func(value,variable)
					
					elif field in ['fixed']:
						func = lambda value,variable: np.array([variable.index(v) if (v in variable) else v for v in value
											if (v in variable) or ((isinstance(v,(int,np.integer)) and v<len(variable)))]).astype(int)
						variable = rhs[dim]
			
						value_keys = func(list(value),variable)
						value_values = [value[k]*info['scale']['y'][dim]/info['scale']['X'][dim][value_keys[i]] 
										for i,k in enumerate(value) if k in variable or isinstance(k,(int,np.integer))]
						
						value = dict(zip(value_keys,value_values))

					elif field in ['cv']:
						if value.get('n_splits') is None:
							value['n_splits'] = n
						elif isinstance(value.get('n_splits'),str):
							value['n_splits'] = {value['n_splits']:n}
						elif isinstance(value.get('n_splits'),(float,np.float)):
							value['n_splits'] = int(n*value['n_splits'])
						elif not isinstance(value.get('n_splits'),(int,np.integer)):
							value['n_splits'] = 2

					info['kwargs'][dim][field] = value

		return

	def postprocess(data,metadata,info,stats,key,typed,label,dim,group,keys,parameters):


		# Copy stats to metadata
		metadata['stats'] = copy.deepcopy(stats)


		# Copy info to metadata
		fields = ['key','keys','groups','indices','operators','scale','rescale','data','rhs','lhs','iloc','distances','refinement']

		for field in fields:
			if field in info[typed][key]:
				metadata['stats'][field] = copy.deepcopy(info[typed][key][field])


		# Copy inherited info to metadata
		fields = ['refinement']

		for field in fields:
			if field in info[info[typed][key]['type']][info[typed][key]['key']]:
				metadata['stats']['%s_inherited'%field] = copy.deepcopy(info[info[typed][key]['type']][info[typed][key]['key']][field])


		X = metadata['stats']['data']['X'][:,:,dim]
		y = metadata['stats']['data']['y'][:,dim]

		scale_X = copy.deepcopy(metadata['stats']['scale']['X'][dim])
		scale_y = copy.deepcopy(metadata['stats']['scale']['y'][dim])

		rhs = copy.deepcopy(metadata['stats']['operators']['X'][dim])
		lhs = copy.deepcopy(metadata['stats']['operators']['y'][dim])		

		indices = metadata['stats']['indices'][dim]
		groups = metadata['stats']['groups'][dim]

		wheregroups = groups==group
		wheredata = np.array([where(index==where(wheregroups))[0] for index in indices[wheregroups[indices]]])
		wheredataset = where(wheregroups[indices])


		assert wheredata.size == wheredataset.size, "Error - mismatched indices and groups for prediction indexing"


		if metadata['stats'].get('rescale') is None:
			metadata['stats']['rescale'] = {}
			metadata['stats']['rescale'].update({k:True for k in ['predict','loss','score','coef_']})
			metadata['stats']['rescale'].update({k:True for k in ['loss','score']})


		if metadata['stats']['rescale']['predict']:
			if X.ndim > 2:
				for i in range(X.shape[2]):
					X[:,:,i] *= invert(scale_X[i],constant=1.0)
			else:
				X *= invert(scale_X,constant=1.0)

			if y.ndim > 1:
				for i in range(y.shape[1]):
					y[:,i] *= invert(scale_y[i],constant=1.0)
			else:
				y *= invert(scale_y,constant=1.0)
		

		if all([metadata['stats'].get(k) is not None for k in ['predict','dim_','size_']]):
			dim_ = np.array(metadata['stats']['dim_']).astype(int)
			Ndim = np.max(dim_)+1

			for k,preds in enumerate(metadata['stats']['predict']):
				label_ = DELIMITER.join([label,str(metadata['stats']['size_'].max()-k)])		
				predictions = np.nan*ones((data.shape[0],*preds.shape[1:]))
				if label_ in data:
					predictions = data[label_].values
				predictions[wheredata] = preds[wheredataset]


				if scale_y.size > 1:
					scale = scale_y[dim_[k]]
				else:
					scale = scale_y


				if metadata['stats']['rescale']['predict']:
					data[label_] = predictions*invert(scale,constant=1.0)
				else:
					data[label_] = predictions


		if all([metadata['stats'].get(k) is not None for k in ['loss','score','dim_','size_']]):
			dim_ = np.array(metadata['stats']['dim_']).astype(int)
			Ndim = np.max(dim_)+1

			for k,(losses,scores) in enumerate(zip(metadata['stats']['loss'],metadata['stats']['score'])):
				label_ = DELIMITER.join([label,str(metadata['stats']['size_'].max()-k)])		

				if scale_y.size > 1:
					scale = scale_y[dim_[k]]
				else:
					scale = scale_y


				if metadata['stats']['rescale']['loss']:
					metadata['stats']['loss'][k] = metadata['stats']['loss'][k]*invert(scale,constant=1.0)
				if metadata['stats']['rescale']['score']:
					metadata['stats']['score'][k] = metadata['stats']['score'][k]*invert(scale,constant=1.0)



		if all([metadata['stats'].get(k) is not None for k in ['coef_','index_','dim_']]):

			r = np.array(rhs)
			dim_ = np.array(metadata['stats']['dim_'][1:]).astype(int)
			Ndim = np.max(dim_)+1 if dim_.size > 0 else 1

			if scale_X.ndim < 2:
				scale_X = np.repeat(scale_X[None,...],Ndim,axis=0)
			if scale_y.size == 1:
				scale_y = np.repeat(scale_y[None,...],Ndim,axis=0)
			elif scale_y.size == 0:
				scale_X = np.repeat(scale_y,Ndim,axis=0)

			if metadata['stats']['rescale']['coef_']:
				coef_ = metadata['stats']['coef_']				
				coef_ = ((invert(scale_y[np.array(metadata['stats']['dim_']).astype(int)],constant=1.0)*coef_.T).T)*(
						  (scale_X[np.array(metadata['stats']['dim_']).astype(int)]))

			else:
				coef_ = metadata['stats']['coef_']

			# Ndim = (max(set(dim_))+1) if len(dim_)>0 else 1

			if r.ndim == 1:
				r = np.repeat(r[None,...],Ndim,0)

			Nrhs = r.shape[1]
			
			index_ = np.array(metadata['stats']['index_'][1:]).astype(int)
			index_null = sorted(arange(Nrhs)[isin(arange(Nrhs),index_,invert=True)],
								key=lambda i,c=coef_[-1]: -i)
			dim_null = zeros(len(index_null)).astype(int)
			ordering_ = [*r[dim_,index_],*r[dim_null,index_null]]
			
			coef_full_ = coef_[0]
			coef_parsimonious_ = coef_[-1]
			coef_ = {o:np.array([c[i] for c in coef_]) for j,_r in enumerate(r) 
						for i,o in enumerate(_r)}

			metadata['stats']['coef_'] = coef_
			metadata['stats']['coef_full_'] = coef_full_
			metadata['stats']['coef_parsimonious_'] = coef_parsimonious_
			metadata['stats']['ordering'] = ordering_
			metadata['stats']['operators'] = np.array(r[0])
			metadata['stats']['size'] = r[0].size
		else:
			metadata['stats']['ordering'] = metadata['stats']['operators']['X'][dim]


		# Remove large fields
		fields = ['data','predict']
		for field in fields:
			metadata['stats'].pop(field);


		return




	# Setup datasets and routines for fitting 
	# Generally each dataset is fit to itself,
	# with priority to fit over prediction
	def setup(data,metadata,info,rhs,lhs,models,parameters):
		def concat(data,key,info,rhs,lhs,models,parameters,metadata):
			keys = info['keys']
			iskeys = len(keys) == 0
			if iskeys:
				info.pop(key)
				return
			values = {}
			fields = {'data':{},'rhs':[],'lhs':[],'groups':[],'model':None}
			for field in fields:
				values[field] = fields[field]

			keys = [k for k in keys if k in data]
			index = keys.index(key) if key in keys else 0
			sizes = [data[k].shape[0] for k in keys]
			size = sum(sizes)

			values['rhs'] = rhs[keys[index]]
			values['lhs'] = lhs[keys[index]]

			if len(keys) > 1:
				values['data']['data'] = pd.concat([data[k] for k in keys],axis=0,ignore_index=True)
			else:
				values['data']['data'] = data[keys[0]]
			values['groups'] = concatenate(tuple(j*ones(i) for j,i in enumerate(sizes)),axis=0).astype(int) 
			values['model'] = models[info['key']] if info['key'] in models else models[keys[index]]				

			values.update({k: copy.deepcopy(metadata[keys[index]].get(k,parameters.get(k))) for k in {**parameters,**metadata[keys[index]]} if k not in values})

			nocopy = ['data']
			for k in values:
				if k in nocopy:
					info[k] = values[k]
				elif k in info:
					if isinstance(values[k],dict):
						info[k].update({u: copy.deepcopy(values[k][u]) for u in values[k] if u not in info[k]})
					else:
						pass
				else:
					info[k] = copy.deepcopy(values[k])

			return
	

		# Check data
		if data is None or len(data)==0:
			return {}

		# Fit info with mandatory keys
		fits = {'fit':['all'],'predict':[]}
		approach = parameters.get('approach','indiv')
		constraints = {}
		constraints['mandatory'] = {'fit':'predict','approach':'fit'}
		constraints['multi'] = {'fit':'approach'}

		# Set info
		info = copy.deepcopy(info)
		if info in [None]:			
			info = {}
			info.update({typed: {
				**{key:{'type':typed, 'key':key,'keys':[key] if key in data else [*data]}
				for key in [*data]},
				**info.get(typed,{}),					
				} 
				for typed in fits if len(fits[typed])>0})

		# Ensure mandatory info exist and if approach is multi, all individual fits are approach type
		ismandatory = all([typed in info for typed in fits if len(fits[typed])>0])	
		if not ismandatory:
			info.update({typed: {
				**{key:{'type':typed,'key':key,'keys':[key] if key in data else [*data]} 
				for key in [*fits[typed]]},
				**info.get(typed,{}),
				}
				for typed in fits if len(fits[typed])>0})

		if approach in ['multi']:
			info.update({typed: {
				**{key:{'type':typed,'key':key,'keys':[key] if key in data else [*data]} 
				for key in [*fits[typed]]},
				**info.get(typed,{}),
				}
				for typed in fits if len(fits[typed])>0})
			info.update({constraints[approach].get(typed,typed): {
				**{k:{**copy.deepcopy(info[typed].pop(k)),
						'type':typed,'key':s,'keys':[key] if key in data else [*data]} 
				for k in [k for k in info[typed] if k not in fits[typed]]
				for s in [k for k in fits[typed]]},
				**{k:v for k,v in info.get(constraints[approach].get(typed,typed),{}).items()}
				}
				for typed in fits if len(fits[typed])>0})


		# Get data,groups,dims for datasets, depending on approach:
		# dataset approaches include individual and multiple datasets
		fields = ['data','keys','rhs','lhs','groups','model']		

		keys = list(set([key for typed in info for key in info[typed]]))

		for typed in list(info):
			for key in list(info[typed]):
				for field in fields:
					if (key in info[typed]) and (field in info[typed][key]) and callable(info[typed][key][field]):
						info[typed][key][field](data,typed,key,info)

		for typed in info:
			for key in info[typed]:
				field = 'key'
				if field not in info[typed][key]:
					info[typed][key][field] = key
				else:
					pass

				field = 'keys'
				if field not in info[typed][key]:
					info[typed][key][field] = [key] if (key in data) else [*data]
				else:
					pass

				concat(data,key,info[typed][key],rhs,lhs,models,parameters,metadata)

		# Order info
		info = {typed:info[typed] for typed in [*fits,*[typed for typed in info if typed not in fits]] if typed in info}

		return info



	# Fitting loop
	# @profile
	def loop(dim,info,metadata,estimator,parameters,verbose=False):
		
		# Fit data for dim

		label_dim = DELIMITER.join([label,typed,str(dim)])
		kwargs_dim = {**parameters,**info[typed][key]['kwargs'][dim]}
		estimator_dim = dim

		for group,_key in enumerate(info[typed][key]['keys']):
			if not isinstance(metadata[_key][typed].get(label_dim),dict):
				metadata[_key][typed][label_dim] = {}

		logger.log(verbose,'Start Dim Fit:\nLHS: %r, dim: %r\ny.shape: %r, X.shape: %r X.rank: %r'%(
				info[typed][key]['operators']['y'][dim],dim,y[:,dim].shape,X[:,:,dim].shape,rank(X[:,:,dim if isinstance(dim,(int,np.integer))else 0],**kwargs_dim)))


		# Set estimator, either creating new base estimator, or using existing estimator
		estimators = {'CrossValidate':CrossValidate,'Stepwise':Stepwise,'OLS':OLS}
		estimator_base = estimators.get(estimator,estimators['OLS'])
		if info[info[typed][key]['type']][info[typed][key]['key']].get('estimator',{}).get(estimator_dim) is None:
			info[typed][key]['estimator'][estimator_dim] = estimator_base(**kwargs_dim)
		else:
			info[typed][key]['estimator'] =  copy.deepcopy(
				info[info[typed][key]['type']][info[typed][key]['key']]['estimator'])
		
		# Get ilocs for each dimension of new estimator based on ilocs for new estimator and 
		# available ilocs associated with each dimension of the reference estimator

		# For each iloc in new estimator for this dimension, to fit data separately within reference estimator, 
		# find dimension of reference estimator that fits with that iloc
		# Currently finds first match of iloc in reference estimator ilocs, however other criteria could be imposed
		# if certain topologies or other factors mean certain reference estimators that do fit at a certain iloc will
		# give better predictions for new estimator

		# ilocs is a dictionary with a key for each new estimator iloc for this dimension, 
		# with values of the reference estimator dimension that fits with that iloc

		ilocs = {}
		for iloc in info[typed][key]['ilocations'][dim]:
			for _dim in info[info[typed][key]['type']][info[typed][key]['key']]['dims']:
				if iloc in info[info[typed][key]['type']][info[typed][key]['key']]['ilocations'][_dim]:
					ilocs[iloc] = _dim
					break



		time_start = timeit.default_timer()

		if typed in ['fit']:
			info[typed][key]['estimator'][estimator_dim].fit(X[:,:,dim],y[:,dim],**kwargs_dim)

		elif typed in ['approach']:
			attrs = {'method':parameters.get('method','update')}
			for attr in attrs:
				if hasattr(info[typed][key]['estimator'],attr):
					setattr(info[typed][key]['estimator'],attr,attrs[attr])				
			info[typed][key]['estimator'][estimator_dim].fit(X[:,:,dim],y[:,dim],**kwargs_dim)

		elif typed in ['update']:
			info[typed][key]['estimator'][estimator_dim].statistics(X[:,:,dim],y[:,dim],None,None,True,**kwargs_dim)


		elif typed in ['predict','interpolate']:
			estimator_ = copy.deepcopy(info[typed][key]['estimator'][estimator_dim])
			_estimator = copy.deepcopy(info[typed][key]['estimator'])

			fields = list(sorted(list(set([field for _dim in _estimator for field in _estimator[_dim].get_stats()])),key=lambda field: list(estimator_.get_stats()).index(field)))

			estimator_.set_stats({field:[] for field in fields})

			_coef_ = {iloc: copy.deepcopy(_estimator[ilocs[iloc]].get_stats()['coef_'])
						for iloc in info[typed][key]['ilocations'][dim]}

			# Get separate predictions for each iloc for the given dim
			for iloc in info[typed][key]['ilocations'][dim]:

				_dim = ilocs[iloc]

				indexes = min([_estimator[_dim].get_stats()[stat].shape[0] 
							for stat in _estimator[_dim].get_stats() if _estimator[_dim].get_stats()[stat] is not None])

				_estimator[_dim].statistics(X=None,y=None,stats=_estimator[_dim].get_stats(),index=None,append=False)

				_fields = [k for k in _estimator[_dim].get_stats() if _estimator[_dim].get_stats()[k] is None]						

				for index in range(indexes):
					coef_ = _coef_[iloc][index]*(
									info[info[typed][key]['type']][info[typed][key]['key']]['scale']['X'][_dim]/
									info[typed][key]['scale']['X'][dim]*
									info[typed][key]['scale']['y'][dim]/
									info[info[typed][key]['type']][info[typed][key]['key']]['scale']['y'][_dim]
									)	
					_estimator[_dim].statistics(
						X[info[typed][key]['ilocations'][dim][iloc],:,dim],
						y[info[typed][key]['ilocations'][dim][iloc],dim],
						_estimator[_dim].get_stats(),
						index=index,
						append=False,
						coef_=coef_,
						**kwargs_dim)

					for field in fields:
						if len(estimator_.get_stats()[field]) < (index+1):
							if field in ['predict']:
								estimator_.get_stats()[field].append(zeros(X.shape[0]))
							else:	
								estimator_.get_stats()[field].append([])

						if field in ['predict']:
							estimator_.get_stats()[field][index][info[typed][key]['ilocations'][dim][iloc]] = _estimator[_dim].get_stats()[field]									
						elif field in _fields:									
							estimator_.get_stats()[field][index].append(_estimator[_dim].get_stats()[field][0])
						else:
							estimator_.get_stats()[field][index].append(_estimator[_dim].get_stats()[field][index])



			for field in fields:
				if field in ['loss']:
					func = lambda stats,sizes,scales,ord=2: np.power((sizes*(np.abs(scales*stats)**ord)).sum(axis=1)/sizes.sum(),1.0/ord)
				elif field in ['score']:
					func = lambda stats,sizes,scales,ord=2: np.power((sizes*(np.abs(scales*stats)**ord)).sum(axis=1)/sizes.sum(),1.0/ord)
				elif field in ['criteria']:
					func = lambda stats,sizes,scales: np.mean(stats,axis=1)
				elif field not in ['coef_','predict']:
					func = lambda stats,sizes,scales: np.mean(stats,axis=1)
				elif field in ['coef_']:
					func = lambda stats,sizes,scales: np.mean(stats,axis=1)
				elif field in ['predict']:
					func = lambda stats,sizes,scales: stats
				else:	
					func = lambda stats,sizes,scales: stats
				sizes = np.array([len(info[typed][key]['ilocations'][dim][iloc])	
						 			for iloc in info[typed][key]['ilocations'][dim]])
				scales = np.array([
							1
				 			for iloc in info[typed][key]['ilocations'][dim]])
				stats = np.array(estimator_.get_stats()[field])	

				estimator_.set_stats({field: func(stats,sizes,scales).astype(stats.dtype)})


			# Check stats are gathered correctly through comparison of prediction and loss
			iteration = 0
			tol = {'predict':1e-15,'interpolate':1e20}[typed]
			eps = np.abs(estimator_.get_stats()['loss'][iteration]-estimator_.get_loss_func()(y[:,dim],X[:,:,dim].dot(estimator_.get_stats()['coef_'][iteration]),**kwargs_dim))

			info[typed][key]['estimator'][estimator_dim] = estimator_


		time_end = timeit.default_timer()	


		# Post process data
		for group,_key in enumerate(info[typed][key]['keys']):
			postprocess(data.get(_key,info[typed][key]['data']['data']),metadata[_key][typed][label_dim],
						info,info[typed][key]['estimator'][dim].get_stats(),
						key,typed,label_dim,dim,group,info[typed][key]['keys'],
						kwargs_dim)

			logger.log(verbose,'Key: %s\nDim: %d/%d\nLoss: %r\nIndex:  %r\nCoef: %s\nTime:  %0.3e s\nDone Dim Fit\n'%(
					_key,dim,len(info[typed][key]['dims']),
					# [l for l in np.atleast_1d(info[typed][key]['estimator'][estimator_dim].loss(X[:,:,dim],y[:,dim],
					# 	coef_=metadata[_key][typed][label_dim]['stats']['coef_full_'],**kwargs_dim))],
					[metadata[_key][typed][label_dim]['stats']['loss'][0]],
					[*[i for i in metadata[_key][typed][label_dim]['stats']['index_'][1:]],
					 *[i for i in range(metadata[_key][typed][label_dim]['stats']['size']-1,-1,-1) if i not in metadata[_key][typed][label_dim]['stats']['index_']]],
					'\n\t'.join(['',*['%s: %r'%(k,metadata[_key][typed][label_dim]['stats']['coef_'][k].tolist()[:10]) for k in list(metadata[_key][typed][label_dim]['stats']['coef_'])[:20]]]),
					time_end-time_start,
					))			
		

		# units = 'MB'
		# fields = ['predict','loss']
		# print('Estimator size: %s %s %d ::: %0.4f %s'%(typed,key,estimator_dim,getsizeof(info[typed][key]['estimator'][estimator_dim],units=units),units))
		# for field in fields:
		# 	print('%s size: %s %s %d: :: %0.4f %s'%(field,typed,key,estimator_dim,getsizeof(info[typed][key]['estimator'][estimator_dim].get_stats().get(field)	,units=units),units))
		# print('Metadata size: %s %s %s ::: %0.4f %s'%(typed,key,label_dim,getsizeof(metadata[key][typed][label_dim],units=units),units))
		# print('Total Metadata size: %s %s ::: %0.4f %s'%(typed,key,getsizeof(metadata[key][typed],units=units),units))

		return


	# Setup info and data

	info = setup(data,metadata,info,rhs,lhs,models,parameters)

	for typed in info:
		for key in info[typed]:


			# Start Fitting
			logger.log(verbose,'Start Total Fit:\nType: %s, Key:%s\nwith\nType: %s, Key: %s'%(typed,key,info[typed][key]['type'],info[typed][key]['key']))

			if any([info[typed][key].get(k) is None for k in ['rhs','lhs']]):
				continue

			# Pre-Process Data
			preprocess(info[typed][key],label,parameters)

			X = info[typed][key]['data']['X']
			y = info[typed][key]['data']['y']

			for group,_key in enumerate(info[typed][key]['keys']):
				if not isinstance(metadata.get(_key),dict):
					metadata[_key] = {}
				if not isinstance(metadata[_key].get(typed),dict):
					metadata[_key][typed] = {}
	
			logger.log(verbose,'LHS: %r, y.shape: %r, X.shape: %r X.rank: %r\n'%(
					info[typed][key]['lhs'],y.shape,X.shape,[rank(X[:,:,dim],**parameters) for dim in info[typed][key]['dims']])) 
					

			# Fit Data

			# Iterate through each dimension of model to fit
			for dim in info[typed][key]['dims']:
				loop(dim,info,metadata,estimator,parameters,verbose=verbose)
			
			logger.log(verbose,'Done Total Fit\n')



	# Check post-processing is correct
	loss_funcs = {'l1':1,'l2':2,'rmse':2,None:2}
	loss_func = parameters.get('loss_func')
	if loss_func in loss_funcs:
		l = loss_funcs[loss_func]
		tol = 1e-12
		dim = 0
		estimator_dim = 0
		iterations = [0,-1]
		fields = ['loss']
		size = {field:{iteration: [] for iteration in iterations} for field in fields}
		scale = {field:{iteration: [] for iteration in iterations} for field in fields}
		value = {field:{iteration: [] for iteration in iterations} for field in fields}
		fields = {field:{iteration: [] for iteration in iterations} for field in fields}
		for typed in info:
			if typed not in ['predict']:
				continue
			keys = [(info[typed][key]['type'],info[typed][key]['key']) for key in info[typed]]
			keys = {key: info[key[0]][key[1]]['keys'] for key in keys}
			keys = {key: list(set([k for k in info[typed] if k in keys[key]])) for key in keys}

			for iteration in iterations:
				for key in keys:
					for field in fields:
						fields[field][iteration].append(info[key[0]][key[1]]['estimator'][estimator_dim].get_stats()[field][iteration])
						size[field][iteration].append(info[key[0]][key[1]]['data']['y'][:,dim].size)
						scale[field][iteration].append(info[key[0]][key[1]]['scale']['y'][dim])
						value[field][iteration].append(info[key[0]][key[1]]['data']['y'][:,dim])

						for k in keys[key]:
							fields[field][iteration].append(info[typed][k]['estimator'][estimator_dim].get_stats()[field][iteration])
							size[field][iteration].append(info[typed][k]['data']['y'][:,dim].size)
							scale[field][iteration].append(info[typed][k]['scale']['y'][dim])
							value[field][iteration].append(info[typed][k]['data']['y'][:,dim])


						if field in ['loss']:
							eps = np.abs((size[field][iteration][0]*(fields[field][iteration][0]/scale[field][iteration][0])**l) - 
											sum([(size[field][iteration][i]*(fields[field][iteration][i]/scale[field][iteration][i])**l) 
								for i in range(1,len(size[field][iteration]))]))
							assert  eps < tol, "Error losses not calculated correctly %0.3e > %0.3e"%(eps,tol)


	
	return		


