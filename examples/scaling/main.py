#!/usr/bin/env python

# Import python modules
import sys,os,glob,copy,itertools,json
import numpy as np
import pandas as pd


import matplotlib

# Import user modules

# Global Variables
DELIMETER='__'
PATH = '../..'

# Import user modules
os.environ['mechanoChemML_SRC_DIR']=PATH
environs = {'mechanoChemML_SRC_DIR': [None,'utility']}
for name in environs:
	path = os.getenv(name)
	assert path is not None, "%s is not defined in the system"%path

	for directory in environs[name]:
		if directory is None:
			directory = path
		else:
			directory = os.path.join(directory,path)
		sys.path.append(directory)

from utility.graph_main import main
from utility.graph_settings import permute_settings
from utility.graph_utilities import mesh,polynomials,scinotation,ncombinations,icombinations
from utility.load_dump import load as loader

np.set_printoptions(linewidth=200)

if __name__ == "__main__":

	# Load command line args
	args = {'settings':{'value':'settings.json','default':{}}}
	args.update({k:{**args[k],'value':v} for k,v in zip(args,sys.argv[1:])})
	args.update({k: {**args[k],'value':loader(args[k]['value'],wr='r',default=args[k]['default'])} for k in args})




	# Booleans for data and plotting
	load = 0
	dump = 1
	plot = 0
	collect = 1


	# Data size and Function order
	p = 1
	d = 1
	N = 6
	N = (2**(np.arange(N,N+1,1))).astype(int)
	K = np.array([5,9])
	O = np.array([1,2,3,4,5,6])

	O = np.array([3])
	K = O+3

	for i,n in enumerate(N):
		for j,k in enumerate(K):
			for l,o in enumerate(O):

				if o>=k:
					continue

				# Data mesh
				n = n+1
				p = p
				k = k
				o = o
				d = d
				N = n**p
				size = ncombinations(p,k,unique=True)-1
				distribution = 'uniform'
				bounds = (-1,1)
				# distribution = 'randommesh'
				# bounds = (0,1)				
				selection = 'unique'
				X = mesh(n=n,d=p,bounds=bounds,distribution=distribution)

				# Non-local derivatives
				order = o
				accuracy = order+1
				unique = True
				format = 'csr'#'array'
				chunk = 0.2
				# chunk = None
				q = ncombinations(p,order,unique=unique)
				operations = ['partial']
				weights = ['stencil']
				adjacency = ['nearest']

				# Basis
				basis = 'taylorseries'
				# basis = 'polynomial'

				# Verbose
				verbose = {
					'boolean':True,
					'sys':True,
					'structure':True,
					'terms':True,
					'model':True,
					'fit':False,
					'analysis':True,
					'plot':True,
				}

				# Directories
				load_cwd = './input'
				dump_cwd = './output'
				load_cwd,dump_cwd = (os.path.abspath(load_cwd),os.path.abspath(dump_cwd)) if not load else (os.path.abspath(dump_cwd),os.path.abspath(dump_cwd))
				files = ['data_%d_%d.csv'%(n,k)]
				# files = ['data_*.csv']

				# Inputs and Outputs
				# inputs = ['x','y','z','w'][:p]	
				inputs = ['x_%d'%(i) for i in range(p)] if p>1 else ['x']
				outputs = ['u','v'][:d]
				terms = ['partial__1__u__x__stencil','partial__1__u__y__stencil','partial__partial__2__u__x__x__stencil__stencil']




				# Mesh refinement
				refinement = [1,int(np.log2(n)) - int(np.log2(o))-1,1]
				# refinement = None

				# iloc = 0 if refinement is None else [0] #True
				iloc = 0 if refinement is None else True
				# iloc = [0] if refinement is None else [0,2,4]
						


				# Polynomial basis
				P,D,alpha,label,derivative,indices = polynomials(X,order=k,derivative=order,intercept_=True,commutativity=unique,
											variables=inputs,selection=selection,
											return_poly=True,return_derivative=True,
											return_coef=True,
											return_polylabel=True,return_derivativelabel=True,return_derivativeindices=True)


				# Terms in function
				model = {'-'.join(['%s_%d'%(y,j) for y,j in zip(inputs,i)]):2*np.random.rand()-1 if (sum(i) in list(range(order+1))) or 1 else 0 for i in itertools.product(range(k+1),repeat=len(inputs)) if sum(i)<=(k)} # only constant, uncoupled linear and quadratic terms

				# print('model')
				# for m in model:
				# 	print(m,model[m])
				# print()
				# exit()
				# print('labels')
				# for m in label:
				# 	print(m,label[m])
				# print()
				# print('derivatives')
				# for m in derivative:
				# 	print(m,derivative[m])
				# exit()
				# print()
				# print('indices')
				# for i,m in enumerate(indices):
				# 	print(m,i)
				# print()	
				# exit()							

				alpha = np.concatenate([alpha[...,None]]*d,axis=-1)
				for m in model:
					i = [u for u in label if m==label[u]]
					if len(i)>0:
						alpha[i[0]] = model[m]
				


				y = P.dot(alpha)
				dydx = D.dot(alpha)


				# Save data as dataframe
				if not load:
					file = 'data_%d_%d.csv'%(n,k)
					path = os.path.join(load_cwd,file)				
					data = np.concatenate([X,y],axis=1)
					columns = [*inputs,*outputs]
					df = pd.DataFrame(data,columns=columns)
					dirname = os.path.dirname(path)
					basename = os.path.basename(path)
					if not os.path.exists(dirname):
						os.mkdir(dirname)
					df.to_csv(path,index=False)


				# Latex Texify
				texargs_texstrings = {}
				texargs_texstrings.update({
					**{x:r'%s'%(x) for x in inputs},
					**{y:r'%s'%(y) for y in outputs},
					})
				# texargs_texstrings.update({
				# 	**{DELIMETER.join([*[p]*j,str(j),y,*x,*[w]*j]):r'\frac{{%s}^{%s} %s}{%s}'%(r'\delta',str(j) if j>1 else '',texargs_texstrings[y],
				# 					r' '.join(['%s %s'%(r'\delta',texargs_texstrings[u]) for u in x]) if len(set(x))>1 else r'{%s} {%s}^{%s}'%(r'\delta',texargs_texstrings[x[0]],str(j) if j>1 else ''))
				# 				for j in range(1,order+1)
				# 				for w in weights
				# 				for p in operations
				# 				for y in outputs
				# 				for x in icombinations(inputs,[j],unique=unique)
				# 			}
				# 	})

				texargs_inputs = {
					**(
					{x:texargs_texstrings[x] for x in inputs} if basis in ['taylorseries','polynomial'] else
					{x:texargs_texstrings[x] for x in terms[1:]}),
					# **{y:texargs_texstrings[y] for y in outputs},		
					}
				texargs_outputs = {
					**({x:texargs_texstrings[x] for x in outputs} if basis in ['taylorseries','polynomial'] else
						{y: texargs_texstrings[y] for y in terms[:1]})
					}
				texargs_terms = {
					**{y:texargs_texstrings[y] for y in inputs},
					**{y:texargs_texstrings[y] for y in outputs},
					# **{y:y for y in terms}

				}


				# Settings
				settings = {
							# Verbose
							**{'%s__verbose'%(k):[verbose[k]] for k in verbose},

							# System directories and files
							'sys__directories__cwd__load':[load_cwd],
							'sys__directories__cwd__dump':[dump_cwd],
							'sys__directories__src__load':[load_cwd],
							'sys__directories__src__dump':[dump_cwd],
							'sys__directories__directories__load':[['']],
							'sys__directories__directories__dump':[['']],
							'sys__label':[None],
							'sys__files__files':[files],							

							'sys__read__data':['r'],
							'sys__read__metadata':['rb'],
							'sys__write__data':['w'],
							'sys__write__metadata':['wb'],
							'sys__labels':[[]],

							# Booleans to turn on/off code functionality
							'boolean__load':[load],
							'boolean__dump':[dump],
							'boolean__plot':[plot],
							'boolean__terms': [1],
							'boolean__model': [1],
							'boolean__fit': [1],
							'boolean__analysis':[1],

							# Graph structure
							'structure__index':[None],
							'structure__parameter':[None],
							'structure__filters':[None],
							'structure__conditions':[None],
							'structure__refinement':[
									{'base':2,
									'n':n,
									'p':p,
									'powers':refinement,
									'settings':[],'keep':False}] if refinement is not None else [None],				
							
							# Terms for model fitting
							'model__inputs':[inputs],
							'model__outputs':[outputs],
							'model__manifold':[inputs],
							'terms__terms':[[  		# Specify derivatives to compute
									  {'function':		y,
										'variable': 	list(x),
										'weight':		[w]*(max(j,1)),
										'adjacency':	[a]*(max(j,1)),
										'accuracy': 	[accuracy]*max(j,1),
										'manifold':	[inputs]*(max(j,1)),				
										'dimension':	[inputs.index(u) for u in x],
										'order':		j,
										'operation':	[p]*(max(j,1)),
									   }
									for j in range(order+1)
									for w in weights
									for a in adjacency
									for p in operations
									for y in outputs
									for x in icombinations(inputs,[j],unique=unique)
									]],
							
							# Model basis and rhs and lhs
							'model__order': [order],
							'model__basis':[basis],
							'model__iloc':[iloc],
							'model__accuracy':[accuracy],							
							'model__sparsity':[10000],							
							'model__strict':[False],
							'model__format':[format],							
							'model__unique':[unique],
							'model__chunk':[chunk],
							'model__normalization':['l2'],
							'model__rhs_lhs':[{
								DELIMETER.join([y,*inputs]):{
												'rhs':[[								
													*inputs,
													# *outputs,
													# *terms[1:]
													# *[DELIMETER.join([*[p]*j,str(j),y,*x,*[w]*j]) 
													# 				for j in range(order+1)
													# 				for w in weights
													# 				for p in operations
													# 				for y in outputs
													# 				for x in icombinations(inputs,[j],unique=unique)
											]],
											'lhs':[y]}
									for y in outputs}] if basis not in ['taylorseries'] else [None],


							# Fit settings
							'fit__estimator':['Stepwise'],
							'fit__kwargs':[{
										'estimator':'OLS',
										'solver':'normallstsq',
										'loss_weights':{'l2':1},
										'score_weights':{'l2':1},
										'complexity_min':q,
										'collect':collect,
										# 'parallel':None,
										# 'n_jobs':6,
												}],
							'fit__info':[{
										# 'fit':{'all': {
										# 	'type':'fit',
										# 	'key':'all',
										# 	'keys':[path_load if not load else path_dump],
										# 	}
										# 	for i,(path_load,path_dump) in enumerate(list(zip([load_cwd],[dump_cwd]))[:1])},									
										# 'fit':{path_load if not load else path_dump: {
										# 	'type':'fit',
										# 	'key':path_load if not load else path_dump,
										# 	'keys':[path_load if not load else path_dump],
										# 	}
										# 	for i,(path_load,path_dump) in enumerate(list(zip([load_cwd],[dump_cwd]))[:1])},
										# 'interpolate':{path_load if not load else path_dump: {
										# 	'type':'fit',
										# 	'key':load_cwd if not load else dump_cwd,
										# 	'keys':[path_load if not load else path_dump],
										# 	'parameters':{'iloc':None},
										# 	}
										# for i,(path_load,path_dump) in enumerate(list(zip([load_cwd],[dump_cwd]))[1:])},	
								}],


							# Analyse results
							'analysis__parameters':[{
								'funcstring':r'u_{%d}(x)'%(k),
								'modelstring':r'u_{%d}(x)'%(order),
								'N':N,
								'n':n,
								'p':p,
								'order':order,
								'q':q,
								'K':k,
								'unique':unique,
								'accuracy':accuracy,
								'inputs':inputs,	
								'outputs':outputs,
								'alpha':alpha,
								'label':label,
								'derivative':dydx,
								'operator':derivative,
								'indices':indices,
								}],							

							# Plot and Texify settings
							'plot__fig': [{'Loss':{},'BestFit':{},'Coef':{}}],
							'plot__axes': [{'Loss':{},'BestFit':{},'Coef':{}}],
							'plot__settings__Loss__ax__set_ylim':[{'ymin':1e-12,'ymax':1e0}],
							'plot__settings__Loss__other__y':[terms[:1]],      
							'plot__settings__BestFit__other__y':[terms[:1]],      
							'plot__settings__BestFit__other__x':[inputs[:1]],      
							'plot__settings__BestFit__other__iterations':[[None,2,1]],      
							'plot__settings__Coef__other__y':[terms[:1]],      
							'texify__texargs__inputs': [texargs_inputs],
							'texify__texargs__outputs': [texargs_outputs],  
							'texify__texargs__terms': [texargs_terms],  
							'texify__texstrings':[texargs_texstrings],
							}



				# Update settings
				settings.update(args['settings']['value'])


				# Permute settings
				settings_grid = permute_settings(settings,_copy=False)
				
				for settings in settings_grid:
					data = {}
					metadata = {}
					main(data,metadata,settings)	


				# Plot scaling results
				from scaling import scaling
				paths = [os.path.join(dump_cwd,'analysis__*.pickle')]
				scaling(paths)