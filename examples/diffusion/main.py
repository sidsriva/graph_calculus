#!/usr/bin/env python

# Import python modules
import sys,glob,copy,itertools,os
import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt


# Import user modules

# Global Variables
CLUSTER = 0
PATH = '../..'

DELIMETER='__'
MAX_PROCESSES = 1
PARALLEL = 0

environs = {'mechanoChemML_SRC_DIR':[None,'utility']}
for name in environs:
	path = os.getenv(name)
	if path is None:
		os.environ[name] = PATH
	path = os.getenv(name)	
	assert path is not None, '%s is not defined in the system'%path

	for directory in environs[name]:
		if directory is None:
			directory = path
		else:
			directory = os.path.join(directory,path)
		sys.path.append(directory)

from utility.graph_main import main
from utility.graph_utilities import mesh,polynomials,scinotation,ncombinations,icombinations
from utility.graph_settings import permute_settings
	




if __name__ == '__main__':

##############################################################
# Choose basic load, dump, plot, usetex settings
	
	load = 0
	dump = 1
	plot = 1
	usetex = 1
	rescale = 1


##############################################################
# Choose model key


	keys = [
		# 'exact',
		'mult_full',
		# 'add_fix_grad',
		# 'add_fix_nograd',
		# 'add_nofix_grad',
		# 'add_nofix_nograd',
		# 'mult_grad',
		# 'mult_nograd',
		]

	for key in keys:

##############################################################
# Choose directories and data files

		Nsample = 16
		Nsamples = range(Nsample)
		# Nsamples = [1,2,3,4,5]
		n_splits = 1 + int(Nsample/10)
		n_splits = 1


		cwd = './input'
		samples = 'Sample%i'			
		file = 'data_c.csv'

		folder = {
			'exact':'ProcessDump/Exact',
			'mult_full':'ProcessDump/Approx_mult_full',
			'add_fix_grad':'ProcessDump/Approx_add_fix_grad',
			'add_fix_nograd':'ProcessDump/Approx_add_fix_nograd',
			'add_nofix_grad':'ProcessDump/Approx_add_nofix_grad',
			'add_nofix_nograd':'ProcessDump/Approx_add_nofix_nograd',
			'mult_grad':'ProcessDump/Approx_mult_grad',
			'mult_nograd':'ProcessDump/Approx_mult_nograd',
			}[key]

		
		cwd = os.path.abspath(os.path.expanduser(cwd))
		directories__load = [samples%(i) for i in Nsamples]
		directories__dump = [os.path.join(samples%(i),folder) for i in Nsamples]
##############################################################


##############################################################
# Choose order and basis type, including selection of powers if doing selective polynomial basis

		order = 2
		derivativeorder = 1
		intercept_ = False
		basis = None

		basis = {
			'exact':'polynomial',
			'mult_full':'polynomial',
			'add_fix_grad':None,
			'add_fix_nograd':None,
			'add_nofix_grad':None,
			'add_nofix_nograd':None,
			'mult_grad':'polynomial',
			'mult_nograd':'polynomial',
		}[key]

		weights = ['stencil']
		adjacency = ['nearest']
		operations = ['partial']
		approach = 'indiv'
		estimators = ['OLS']

##############################################################
	


##############################################################
# Choose lhs as terms[0] and if derivative dy/dx, include appropriate x in inputs and y in outputs
# Choose rhs as terms[1:]
# These terms will be used to calculate derivatives in terms__terms and configure the model__rhs_lhs
# 
		inputs = ['Phi_1P','Time']
		outputs = ['TE','Phi_1P']			
		terms = {
			'exact':	[
				#'DiffVal',
				'partial__1__Phi_1P__Time__stencil',
				# 'partial__1__TE__Phi_1P__stencil',
				'Phi_1P','Phi_2P','Phi_3P','Phi_4P','Phi_5P',
				# 'Phi_1M','Phi_2M','Phi_3M','Phi_4M','Phi_5M',
				'LapC_P',
				# 'GradE_P',
				#'LanE_P','LapC_P','dLan_P',
				# 'GradE_M',
				#'LanE_M','LapC_M','dLan_M'
				],
			'mult_full':	[
				#'DiffVal',
				'partial__1__Phi_1P__Time__stencil','partial__1__TE__Phi_1P__stencil',
				'Phi_1P','Phi_2P','Phi_3P','Phi_4P','Phi_5P',
				'Phi_1M','Phi_2M','Phi_3M','Phi_4M','Phi_5M',
				'GradE_P',
				'LanE_P','LapC_P','dLan_P',
				'GradE_M',
				'LanE_M','LapC_M','dLan_M'
				],
			'add_fix_grad':	[
				#'DiffVal',
				'partial__1__Phi_1P__Time__stencil','partial__1__TE__Phi_1P__stencil',
				'Phi_1P','Phi_2P','Phi_3P','Phi_4P','Phi_5P',
				'Phi_1M','Phi_2M','Phi_3M','Phi_4M','Phi_5M',
				'GradE_P',
				#'LanE_P','LapC_P','dLan_P',
				'GradE_M',
				#'LanE_M','LapC_M','dLan_M'
				],
			'add_fix_nograd':	[
				#'DiffVal',
				'partial__1__Phi_1P__Time__stencil','partial__1__TE__Phi_1P__stencil',
				'Phi_1P','Phi_2P','Phi_3P','Phi_4P','Phi_5P',
				'Phi_1M','Phi_2M','Phi_3M','Phi_4M','Phi_5M',
				'GradE_P',
				#'LanE_P','LapC_P','dLan_P',
				'GradE_M',
				#'LanE_M','LapC_M','dLan_M'
				],
			'add_nofix_grad':	[
				#'DiffVal',
				'partial__1__Phi_1P__Time__stencil','partial__1__TE__Phi_1P__stencil',
				'Phi_1P','Phi_2P','Phi_3P','Phi_4P','Phi_5P',
				'Phi_1M','Phi_2M','Phi_3M','Phi_4M','Phi_5M',
				'GradE_P',
				#'LanE_P','LapC_P','dLan_P',
				'GradE_M',
				#'LanE_M','LapC_M','dLan_M'
				],
			'add_nofix_nograd':	[
				#'DiffVal',
				'partial__1__Phi_1P__Time__stencil','partial__1__TE__Phi_1P__stencil',
				'Phi_1P','Phi_2P','Phi_3P','Phi_4P','Phi_5P',
				'Phi_1M','Phi_2M','Phi_3M','Phi_4M','Phi_5M',
				# 'GradE_P',
				#'LanE_P','LapC_P','dLan_P',
				# 'GradE_M',
				#'LanE_M','LapC_M','dLan_M'
				],
			'mult_grad':	[
				#'DiffVal',
				'partial__1__Phi_1P__Time__stencil','partial__1__TE__Phi_1P__stencil',
				'Phi_1P','Phi_2P','Phi_3P','Phi_4P','Phi_5P',
				'Phi_1M','Phi_2M','Phi_3M','Phi_4M','Phi_5M',
				'GradE_P',
				#'LanE_P','LapC_P','dLan_P',
				'GradE_M',
				#'LanE_M','LapC_M','dLan_M'
				],
			'mult_nograd':	[
				#'DiffVal',
				'partial__1__Phi_1P__Time__stencil','partial__1__TE__Phi_1P__stencil',
				'Phi_1P','Phi_2P','Phi_3P','Phi_4P','Phi_5P',
				'Phi_1M','Phi_2M','Phi_3M','Phi_4M','Phi_5M',
				# 'GradE_P',
				#'LanE_P','LapC_P','dLan_P',
				# 'GradE_M',
				#'LanE_M','LapC_M','dLan_M'
				],
		}[key]

		# Number of inputs and outputs
		m = len(terms)-1
		d = len(outputs)


		included_terms = {
			'exact':[],
			'mult_full':[],
			'add_fix_grad':['partial__1__TE__Phi_1P__stencil'],
			'add_fix_nograd':['partial__1__TE__Phi_1P__stencil'],
			'add_nofix_grad':['partial__1__TE__Phi_1P__stencil'],
			'add_nofix_nograd':['partial__1__TE__Phi_1P__stencil'],
			'mult_grad':[],
			'mult_nograd':[],
			}[key]

		fixterms = {
			'exact':{},
			'mult_full':{},
			'add_fix_grad':{'partial__1__TE__Phi_1P__stencil':-1},
			'add_fix_nograd':{},
			'add_nofix_grad':{},
			'add_nofix_nograd':{},
			'mult_grad':{},
			'mult_nograd':{},
			}[key]

		selection = {
			'exact':np.eye(m, dtype=int),# np.array([[1,0,0,0,0,0],[0,1,0,0,0,0], [0,0,1,0,0,0], [0,0,0,1,0,0], [0,0,0,0,1,0], [0,0,0,0,0,1]]),
			'mult_full':np.vstack((np.eye(m, dtype=int),np.hstack((np.ones([m-1,1], dtype=int),np.eye(m-1, dtype=int))))),
			'add_fix_grad':None,
			'add_fix_nograd':None,
			'add_nofix_grad':None,
			'add_nofix_nograd':None,
			'mult_grad': np.vstack((np.eye(m, dtype=int),np.hstack((np.ones([m-1,1], dtype=int),np.eye(m-1, dtype=int))))),
			'mult_nograd':np.vstack((np.eye(m, dtype=int),np.hstack((np.ones([m-1,1], dtype=int),np.eye(m-1, dtype=int))))),
		 }[key]


		# Total number of terms
		q = m if selection is None else selection.shape[0]

##############################################################
# Set settings
	

		
		

		# Latex Texify
		texargs_texstrings = {}
		texargs_texstrings.update({
			**{x:r'%s'%(x) for x in inputs},
			**{y:r'%s'%(y) for y in outputs},
			**{
				'Time':r't',
				**{'Phi_%dP'%(i):r'\varphi_{%s{}}'%('%d'%i if i>1 else '') for i in range(1,5+1)},
				# **{'Phi_%dM'%(i):r'\bar{\varphi}_{%s{}}'%('%d'%i if i>1 else '') for i in range(1,5+1)},
				**{'Phi_%dM'%(i):r'{\varphi}_{%s{-}}'%('%d'%i if i>1 else '') for i in range(1,5+1)},
				'TE':r'\Psi',
				# 'GradE_P':r'\varphi_{\scriptscriptstyle{|\nabla c|^2_{+}}}',
				'GradE_P':r'\varphi_{\scriptscriptstyle{\nabla_2 c_{+}}}',
				'LanE_P':r'{F_{+}}',
				'LapC_P':r'\varphi_{\scriptscriptstyle{\nabla^2_{+}}}',
				'dLan_P':r'{F^{\prime}_{+}}',
				# 'GradE_M':r'\bar{\varphi}_{\scriptscriptstyle{\nabla_2 c_{+}}}',				
				'GradE_M':r'{\varphi}_{\scriptscriptstyle{\nabla_2 c_{-}}}',				
				# 'LanE_M':r'{\bar{F}_{+}}',
				'LanE_M':r'{{F}_{-}}',
				# 'LapC_M':r'\bar{\varphi}_{\scriptscriptstyle{\nabla^2_{+}}}',				
				'LapC_M':r'{\varphi}_{\scriptscriptstyle{\nabla^2_{-}}}',				
				# 'dLan_M':r'{\bar{F}^{\prime}_{+}}',
				'dLan_M':r'{{F}^{\prime}_{-}}',
				'DiffVal':r'\frac{\partial \Psi}{\partial\varphi_{{}}} + \frac{\partial \varphi_{{}}}{\partial t}',
				}
			})
		texargs_texstrings.update({
			**{DELIMETER.join([*[o]*j,str(j),y,*x,*[w]*j]):r'\frac{{%s}^{%s} %s}{%s}'%(p,str(j) if j>1 else '',texargs_texstrings[y],
							r' '.join(['%s %s'%(p,texargs_texstrings[u]) for u in x]) if len(set(x))>1 else r'{%s} {%s}^{%s}'%(p,texargs_texstrings[x[0]],str(j) if j>1 else ''))
						for j in range(1,derivativeorder+1)					
						for w in weights
						for o,p in zip(['delta','partial'],[r'\delta',r'\partial'])
						for y in outputs
						for x in icombinations(inputs,[j],unique=True)
					}
			})

		texargs_texstrings.update({os.path.join(cwd,directory__load if not load else directory__dump):r'{{\mathcal{D}}_{%d}}'%(i) for i,(directory__load,directory__dump) in enumerate(zip(directories__load,directories__dump))})



		texargs_inputs = {
			**(
			{x:texargs_texstrings[x] for x in inputs} if basis in ['taylorseries'] else
			{x:texargs_texstrings[x] for x in terms[1:]}),
			# **{y:texargs_texstrings[y] for y in outputs},		
			}
		texargs_outputs = {
			**({x:texargs_texstrings[x] for x in outputs} if basis in ['taylorseries'] else
				{y: texargs_texstrings[y] for y in terms[:1]})
			}
		texargs_terms = {
			# **{y:texargs_texstrings[y] for y in terms},
		}	


		
		settings = {
			#Set Paths
			'sys__directories__cwd__load':[cwd],
			'sys__directories__cwd__dump':[cwd],
			'sys__directories__src__load':[cwd],
			'sys__directories__src__dump':[cwd],
			'sys__directories__directories__load':[directories__load],
			'sys__directories__directories__dump':[directories__dump],
			'sys__files__files':[[file]],
			'sys__label':[None],

			'sys__read__data':['r'],
			'sys__read__metadata':['rb'],
			'sys__write__data':['w'],
			'sys__write__metadata':['wb'],
			'sys__labels':[[]], #To change output filenames
			
			#Set Flags
			'boolean__load':[load],
			'boolean__dump':[dump],
			'boolean__verbose':[1],
			'boolean__texify':[1],
			'boolean__display':[1],
			'boolean__plot':[plot],
			
			

			# Graph structure settings
			'structure__index':[None],
			'structure__seed':[1234556789],
			'structure__filters':[None],
			'structure__conditions':[None],
			'structure__refinement':[None],
			
			'structure__functions': [[
				{'func':lambda df: df['partial__1__TE__Phi_1P__stencil'] + df['partial__1__Phi_1P__Time__stencil'], 'labels':'DiffVal'} , 
			]],
			# 'structure__functions': [[
			# 	{'func':lambda df:0.001*df['Energy'] + 1*df['Potential'], 'labels':'TE'} , 
			# 	{'func':lambda df:pow(df['Potential'],-1), 'labels':'Pinv'}, 
			# 	{'func':lambda df: df['Phi_1P']- 	 df['Phi_0P'] , 'labels':'Phi_1P0'}, 
			# 	{'func':lambda df: pow(df['Phi_1P'] - df['Phi_0P'] ,-1), 'labels':'Phi_1P0_Inv'}
			# 	]],			

			#Set Model info
			'model__basis':[basis],
			'model__order':[order],
			'model__p':[1],
			'model__intercept_':[intercept_],
			'model__inputs':[inputs],
			'model__outputs':[outputs],
			'model__selection':[selection],		
			'model__normalization':['l2'],
			'model__format':['csr'],									
			'model__sparsity':[100],									
			'model__unique':[True],
			'model__chunk':[1],
			# 'terms__terms': [[
			# 	{'function':y,
			# 	'variable': list(x),
			# 	'weight':[w]*j, 
			# 	'adjacency':[a]*j, #(symmetric finite difference like scheme),
			# 	'manifold':[inputs]*j, #(or could specify 't' if the vertices are strictly defined by time and adjacency is defined by that, and not per say euclidean distance in 'u' space)
			# 	'dimension':[inputs.index(u) for u in x],
			# 	'order':j,
			# 	'operation':[o]*j,
			# 	}
			# 	for j in range(1,derivativeorder+1)
			# 	for w in weights
			# 	for a in adjacency
			# 	for o in operations
			# 	for y in outputs
			# 	for x in icombinations(inputs,[j],unique=unique)
			# 	]],
			'terms__terms':[[
				{'function':'TE',
				'variable': ['Phi_1P'],
				'weight':['stencil'], 
				'adjacency':['nearest'], #(symmetric finite difference like scheme),
				'manifold':[['Phi_1P']], #(or could specify 't' if the vertices are strictly defined by time and adjacency is defined by that, and not per say euclidean distance in 'u' space)
				'accuracy': [2],		
				'dimension':[0],
				'order':1,
				'operation':['partial'],
				},
				
				{'function':'Phi_1P',
				'variable': ['Time'],
				'weight':['stencil'], 
				'adjacency':['backward_nearest'], #(symmetric finite difference like scheme),
				'manifold':[['Time']], #(or could specify 't' if the vertices are strictly defined by time and adjacency is defined by that, and not per say euclidean distance in 'u' space)
				'accuracy': [1],						
				'dimension':[0],				
				'order':1,
				'operation':['partial'],
				}		
				]],

			'model__rhs_lhs': [{
				'model_label':{
					'lhs': [y],
					'rhs':[[
					*terms[1:]
						]],
				} for y in terms[:1]}],

			# Fit settings
			'fit__verbose': [0],
			'fit__kwargs__estimator': ['RidgeCV'], # type of linear fitting estimator
			'fit__kwargs__cv': [{'cv':'KFolder','n_splits':n_splits,'random_state':1235,'test_size':0.02}], # cross validation settings
			'fit__kwargs__alpha': [[-9,-2, 10]],	# range of regression parameters to sweep over during cross validation [[start,end,NumPoints]]] means NumPoints between 10^-start to 10^-end
			'fit__kwargs__alpha_': [1e-5],	# single regression parameter to use when not doing cross validation
			'fit__kwargs__alpha_zero': [0], # include 0 in alpha cross validation search
			'fit__kwargs__solver': ['svd'], # direct linear solver used (functions found under estimator.py Solver class)
			'fit__kwargs__included': [included_terms] , #To stop included_terms from dropping
			'fit__kwargs__fixed':[fixterms] , #To fix certain values
			'fit__info':[{
				'fit':{'all': {
					'type':'fit',
					'key':'all',
					'keys':[os.path.join(cwd,directory__load) if not load else os.path.join(cwd,directory__dump) for directory__load,directory__dump in zip(directories__load,directories__dump)],
						 },
				 },
				'predict':{os.path.join(cwd,directory__load) if not load else os.path.join(cwd,directory__dump): {
					'type':'fit',
					'key':'all',
					'keys':[os.path.join(cwd,directory__load) if not load else os.path.join(cwd,directory__dump)],
						  }
					for directory__load,directory__dump in zip(directories__load,directories__dump)}
				}
				],

			# Plot Settings
			'texify__usetex':[usetex],
			'texify__texargs__inputs': [texargs_inputs],
			'texify__texargs__outputs': [texargs_outputs],  
			'texify__texargs__terms': [texargs_terms],  
			'texify__texstrings':[texargs_texstrings],
			
			'texify__texargs__bases':[{'monomial':1,'polynomial':0,'taylorseries':1,**{basis:1}}],
			'texify__texargs__selection': [selection],

			# 'plot__fig': [{'Loss':{},'BestFit':{},'Coef':{}}],
			# 'plot__fig': [{'Loss':{},'BestFit':{}}],
			# 'plot__fig': [{'BestFit':{},'Coef':{}}],
			# 'plot__fig': [{'Loss':{}}],
			'plot__fig': [{'BestFit':{}}],
			# 'plot__fig': [{'Coef':{}}],
			# 'plot__fig': [{}],
			'plot__settings__BestFit__other__x':[['Time']],
			'plot__settings__BestFit__other__iterations':[[None,10,5,3,2]],
			# 'plot__settings__BestFit__other__iterations':[[]],
			'plot__settings__BestFit__ax__set_xlim':[{'xmin':-0.1,'xmax':5.6}],
			'plot__settings__BestFit__ax__set_xticks':[{'ticks':[0,2,4]}],
			'plot__settings__BestFit__ax__set_yticks':[{'ticks':[2e-2,4e-2,6e-2,8e-2]}],
			'plot__settings__BestFit__ax__set_ylim':[{'ymin':0,'ymax':9e-2}],

			# 'plot__settings__BestFit__other__iterations':[[]],
			'plot__settings__Coef__ax__grid':[{'b':True,'alpha':0.6,'which':'major'}],			
			'plot__settings__Coef__style__layout':[{'nrows':q//9,'ncols':q//(q//9)+1} if q>9 else {'nrows':m//3,'ncols':3}],
			'plot__settings__Coef__style__share__ax__set_xlabel':[{'xlabel':'bottom'}],
			'plot__settings__Coef__fig__set_size_inches':[{'w':40,'h':35}],
			'plot__settings__Coef__fig__subplots_adjust':[{'wspace':0.75,'hspace':0.5}],
			# 'plot__settings__Coef__fig__tight_layout':[{'pad':300,'h_pad':200,}],
			'plot__settings__Loss__ax__plot__label': ['%s'],
			'plot__settings__Coef__ax__set_title':[{'pad':50}],
			'plot__settings__Loss__ax__set_ylim':[{'ymin':1e-7,'ymax':1e-1}],
			'plot__settings__Loss__ax__grid':[{'b':True,'alpha':0.6,'which':'both'}],

			'plot__settings__Loss__ax__legend':[{
				'title_fontsize': 32,
				'prop': {'size': 35},
				'markerscale': 1.2,
				'handlelength': 3,
				'framealpha': 0.8,
				# 'loc': 'lower right',
				'loc': 'upper left',
				'ncol': 2,
				# 'zorder':100,
				}],

			'plot__retain__fig__Loss':[True],
			'plot__retain__fig__BestFit':[False],
			'plot__retain__fig__Coef':[False],
			# 'plot__retain__key__Loss':[True],
			# 'plot__retain__label__Loss':[True],
			# 'plot__retain__dim__Loss':[True],
			'plot__rescale':[rescale],
			'plot__first_last':[True],
			}



		settings_grid = permute_settings(settings,_copy = True)

		data = {}
		metadata = {}
		main(data,metadata,settings_grid[0])
