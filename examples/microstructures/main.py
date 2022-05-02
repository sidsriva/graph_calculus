#!/usr/bin/env python

# Import python modules
import sys,os,glob,copy,itertools,json
import numpy as np


# Logging
import logging
log = 'warning'
logger = logging.getLogger(__name__)
logger.setLevel(getattr(logging,log.upper()))


# Global Variables
CLUSTER = 0
PATH = '../..'


DELIMETER='__'
MAX_PROCESSES = 1
PARALLEL = 0

# Import user modules
# export mechanoChemML_SRC_DIR=/home/matt/files/um/code/mechanochem-ml-code
environs = {'mechanoChemML_SRC_DIR':[None,'utiilty']}
for name in environs:
	path = os.getenv(name)
	if path is None:
		os.environ[name] = PATH
	path = os.getenv(name)	
	assert path is not None, "%s is not defined in the system"%path

	for directory in environs[name]:
		if directory is None:
			directory = path
		else:
			directory = os.path.join(path,directory)
		sys.path.append(directory)

from utility.graph_main import main
from utility.graph_settings import permute_settings
from utility.dictionary import _get




def select(data,typed,key,info):

		from sklearn.cluster import KMeans
		import scipy.cluster


		if typed not in info or key not in info[typed] or len(data)==1:
			return

		old = list(data)
		constraints = {'fit':'predict'}
		labels = ['E_11','E_22']
		values = {k:data[k][labels].sum(axis=1).mean(axis=0) for k in data}
		values = np.array([values[v] for v in values]).reshape((-1,1))

		n = len(data)
		
		# n_clusters = [n//i for i in range(n,0,-1) if  (n/i==n//i) and i<n]
		# if len(n_clusters)==0:
		# 	n_clusters = n
		# else:
		# 	n_clusters = n_clusters[1]


		for n_clusters in range(n,0,-1):
			km = KMeans(n_clusters=n_clusters)
			estimator = km.fit(values)
			clusters = {j: [old[i] for i in np.where(km.labels_ == j)[0]] for j in range(km.n_clusters)}
			size_clusters = {j: len(clusters[j]) for j in clusters}
			new = {'%d'%(j): {'key':'%d'%(j),'keys':clusters[j]} for j in clusters}
			if min([size_clusters[j] for j in clusters]) < 2:
				continue
			else:
				break

		# # new = {DELIMETER.join(old[i:i+r]): {'keys':old[i:i+r]} for i in range(0,n,r)}
		# new = {'%d'%(j): {'key':'%d'%(j),'keys':old[i:i+r]} for j,i in enumerate(range(0,n,r))}



		info_typed_key = info[typed][key]	
		
		for k in old:
			if constraints.get(typed,typed) not in info:
				info[constraints.get(typed,typed)] = {}
			info[constraints.get(typed,typed)][k] = info[typed].pop(k)
			info[constraints.get(typed,typed)][k]['key'] = [s for s in new for t in new[s]['keys'] if t==k][0]
			info[constraints.get(typed,typed)][k]['keys'] = [k]
			
		info[typed].update({k:{**info_typed_key,**new[k]} for k in new})

		return


def sampler(key=None,typed=None,variables={},**kwargs):
	def combinations(iterables,indices,default,N,stop=None):
		isstop = stop is not None
		for i,iterable in enumerate(iterables):
			if isstop and (i+1)>=stop:
				iterables.close()
			_iterable = [default]*N
			for k,j in enumerate(iterable):
				_iterable[indices[k]] = j
			yield _iterable

	if key in [None]:
		def func(I=None,**kwargs):
			return I
		return func
	elif key in ['psi']:
		def func(I=[],**kwargs):
			return (i for i in I) if I is not None else I
		return func
	elif key in ['vol']:

		if typed in ['physics']:

			# Sets of allowed variables of each independent class of variable
			isvariables = {}
			isvariables_list = {}
			isvariables_list['order'] = None
			isvariables_list['topological'] = None
			isvariables_list['derivatives'] = set((y 
							for w in [	
							[
								['%s%s'%(y,'%s%s'%('' if x in [''] else '_',x)) 
								for y in variables['chemical']['label'][typed] 
								for x in variables['chemical']['dimension'][typed][y] if y in ['vol']],
								['%s%s'%(y,'%s%s'%('' if x in [''] else '_',x)) 
								for y in variables['mechanical']['label'][typed] 
								for x in variables['mechanical']['dimension'][typed][y]]
							],
							[
								['%s%s'%(y,'%s%s'%('' if x in [''] else '_',x)) 
								for y in variables['chemical']['label'][typed] 
								for x in variables['chemical']['dimension'][typed][y] if y in ['vol']],
								['%s%s'%(y,'%s%s'%('' if x in [''] else '_',x)) 
								for y in variables['chemical']['label'][typed] 
								for x in variables['chemical']['dimension'][typed][y] if y in ['vol']]
							],
							[
								['%s%s'%(y,'%s%s'%('' if x in [''] else '_',x)) 
								for y in variables['chemical']['label'][typed] 
								for x in variables['chemical']['dimension'][typed][y] if y in ['vol']],
								['%s%s'%(y,'%s%s'%('' if x in [''] else '_',x)) 
								for y in variables['mechanical']['label'][typed] 
								for x in variables['mechanical']['dimension'][typed][y]],
								['%s%s'%(y,'%s%s'%('' if x in [''] else '_',x)) 
								for y in variables['mechanical']['label'][typed] 
								for x in variables['mechanical']['dimension'][typed][y]],								
							],
							[
								['%s%s'%(y,'%s%s'%('' if x in [''] else '_',x)) 
								for y in variables['chemical']['label'][typed] 
								for x in variables['chemical']['dimension'][typed][y] if y in ['vol']],
								['%s%s'%(y,'%s%s'%('' if x in [''] else '_',x)) 
								for y in variables['chemical']['label'][typed] 
								for x in variables['chemical']['dimension'][typed][y] if y in ['vol']],								
								['%s%s'%(y,'%s%s'%('' if x in [''] else '_',x)) 
								for y in variables['mechanical']['label'][typed] 
								for x in variables['mechanical']['dimension'][typed][y]]
							],
							[
								['%s%s'%(y,'%s%s'%('' if x in [''] else '_',x)) 
								for y in variables['chemical']['label'][typed] 
								for x in variables['chemical']['dimension'][typed][y] if y in ['vol']],
								['%s%s'%(y,'%s%s'%('' if x in [''] else '_',x)) 
								for y in variables['chemical']['label'][typed] 
								for x in variables['chemical']['dimension'][typed][y] if y in ['vol']],
								['%s%s'%(y,'%s%s'%('' if x in [''] else '_',x)) 
								for y in variables['chemical']['label'][typed] 
								for x in variables['chemical']['dimension'][typed][y] if y in ['vol']]								
							],																					
							]
						# for u in itertools.product(*w,repeat=1)
						for u in list(set([tuple(sorted(x)) for x in itertools.product(*w,repeat=1)]))
						for x in (itertools.permutations(u,r=len(u)) if 0 else [u])
						for y in [
								DELIMETER.join([*['partial' for i in range(o)],str(o),y,*x,*[w for i in range(o)]])
								for o in [len(x)]									 		
								for w in kwargs.get('weight',['stencil'])
								for y in ['%s%s'%(y,'%s%s'%('' if x in [''] else '_',x)) 
									for y in variables['energy']['label'][typed] 
									for x in variables['energy']['dimension'][typed][y]
									if x in ['tot']
									]
								]
						)
						)

			# Boolean functions whether variables of each class are allowed
			for l in ['order','topological','derivatives']:
				isvariables[l] = lambda x,v=isvariables_list[l],*args,**kwargs: ((v is None) or (x in v))
		else:
			isvariables = None



		# Allowed indices for each variable in each type
		indices = {}

		indices.update({l:{
			**{k:list(range(0,2+1)) for k in ['%s%s'%(y,'%s%s'%('' if x in [''] else '_',x)) 
					for y in variables['mechanical']['label'][typed] 
					for x in variables['mechanical']['dimension'][typed][y]]},	
			**{k:list(range(0,2+1)) for k in ['%s%s'%(y,'%s%s'%('' if x in [''] else '_',x)) 
					for y in variables['chemical']['label'][typed] 
					for x in variables['chemical']['dimension'][typed][y]
					if y in ['vol']] if (isvariables is None or 1 or isvariables[l](k))},
			} for l in ['order']})
		indices.update({l:{
			**{k:list(range(0,1+1)) for k in ['%s%s'%(y,'%s%s'%('' if x in [''] else '_',x)) 
					for y in variables['chemical']['label'][typed] 
					for x in variables['chemical']['dimension'][typed][y]
					if y in ['len','number']] if (isvariables is None or 1 or isvariables[l](k))},
			} for l in ['topological']})		


		indices.update({l: {
				**{k: list(range(0,1+1)) for k in [DELIMETER.join([*['partial' for i in range(o)],str(o),y,DELIMETER.join(x),*[w for i in range(o)]])
					for o in range(2,3+1)									 		
					for w in kwargs.get('weight',['stencil'])	
					for y in ['%s%s'%(y,'%s%s'%('' if x in [''] else '_',x)) 
							for y in variables['energy']['label'][typed] 
							for x in variables['energy']['dimension'][typed][y]
							if x in ['tot']
							]
					for x in ([x for x in itertools.product([
					# for x in set([tuple(sorted(x)) for x in itertools.product([
							*['%s%s'%(y,'%s%s'%('' if x in [''] else '_',x)) 
							for y in variables['chemical']['label'][typed] 
							for x in variables['chemical']['dimension'][typed][y] if y in ['vol']],
							*['%s%s'%(y,'%s%s'%('' if x in [''] else '_',x)) 
							for y in variables['mechanical']['label'][typed] 
							for x in variables['mechanical']['dimension'][typed][y]]
							],repeat=o)])
					] if (isvariables is None or 1 or isvariables[l](k))},
					} for l in ['derivatives']})


		# Booleans whether each variable is allowed across all classes and variables 
		indices['all'] = {k: isvariables is None or isvariables[l](k) for l in indices for k in indices[l]}

		# Index of allowed variables across all classes and variables
		allowed = [i for i,k in enumerate(indices['all']) if indices['all'][k]]

		# Total number of variables and total allowed variables
		Nvariables = len(indices['all'])
		Nallowed = len(allowed) 


		# Index of each variable of each class i.e) labels['derivative'] = {dudx:0, dudy:1, ... d^3u/dxdydz:k-1}}
		labels = {l: {k:i for i,k in enumerate([k for k in indices[l] if isvariables is None or isvariables[l](k)])} for l in indices if l not in ['all']}

		labels['all'] = {}
		n = -1
		for i,l in enumerate([l for l in labels if l not in ['all']]):
			for j,k in enumerate(labels[l]):
				n += 1
				labels['all'][k] = n

		# Groups of allowed variables, with index of variable within group i.e) types['differential'] = {'derivative':{dudx:0, dudy:1, ... d^3u/dxdydz:k-1}}
		types = {}
		types['differential'] = {l: {k:labels[l][k] for k in labels[l] if k.startswith('partial')} for l in indices}
		types['algebraic'] = {l: {k:labels[l][k] for k in labels[l] if not k.startswith('partial')} for l in indices}
		types['mech'] = {l: {k:labels[l][k] for k in labels[l] if any([k.startswith(l) for l in variables['mechanical']['label'][typed]])} for l in indices}
		types['chem'] = {l: {k:labels[l][k] for k in labels[l] if any([k.startswith(l) for l in variables['chemical']['label'][typed]])}  for l in indices}
		types['chem_geometry'] = {l: {k: labels[l][k] for k in ['%s%s'%(y,'%s%s'%('' if x in [''] else '_',x))
					for y in ['len','number'] 
					for x in variables['chemical']['dimension'][typed][y]] if k in labels[l]}  for l in indices}
		types['mech_chem'] = {l: {k: labels[l][k] for k in [*types['mech'][l],*[k for k in types['chem'][l] if k.startswith('vol')]]} for l in indices}
		

		# Rules whether set of indices inds for allowed variables with class label are allowed
		def rules(inds,label):
			def issum(inds,labels):
				return sum([inds[labels[y]] for y in labels])
			
			if label in ['order']:
				conditions = [
					(issum(inds,labels[label])>=1),
					(issum(inds,types['mech_chem'][label])<=2),
				]
			elif label in ['topological']:
				conditions = [
					(issum(inds,types['chem_geometry'][label])<=1),
				]
			elif label in ['derivatives']:
				conditions = [
					(issum(inds,types['differential'][label])<=1),
				]
			elif label in ['all']:
				inds = sum((*inds,),())
				conditions = [
					(issum(inds,labels[label])>=1),				
					((issum(inds,{k: labels[label][k] for k in types['differential']['derivatives']}) + 
					  issum(inds,{k: labels[label][k] for k in types['chem_geometry']['topological']})) 	<=1),
				]
			return all(conditions)
		

		# Default index
		default = 0	
	
		# All combinations of indices of allowed variables per class as per rules for allowed indices
		if isvariables is not None:
			combinations_variables = {l: filter(lambda iterable,label=l:rules(iterable,label),
							itertools.product(*[indices[l][k] for k in indices[l] if isvariables is None or isvariables[l](k)]
							)) for l in indices if l not in ['all']}
		else:
			combinations_variables = {l: [] for l in indices if l not in ['all']}

		# All combinations of indices of allowed variables across all classes
		combinations_allowed = list([sum((*y,),()) for y in filter(lambda iterable,label='all':rules(iterable,label),itertools.product(*[combinations_variables[l] for l in combinations_variables]))])


		def func(I=None,**kwargs):
			return combinations(combinations_allowed,allowed,default,Nvariables,stop=None)
		
		return func








if __name__ == "__main__":


	_settings = {k:{} for k in ['all','key','default','sys','main','grid']}


	_defaults = {
		'sys': {
			'main__info':['fit'],
			'main__dim':[2],
			'main__keys':[
				{'vol':{'type':'physics','model':'vol'},
				 },
				 ],
			'main__groups':[[]],
			'sys__directories__cwd__load':['input'],
			'sys__directories__cwd__dump':['output'],
			'sys__directories__directories__load':[['data']],
			'sys__directories__directories__dump':[['data']],
			'sys__files__files':[['data.csv']],
			'structure__samples':[None],
			'structure__filters__passes':[3],
			'model__order':[3],
			'model__weight':['stencil'],
			'model__adjacency':['nearest'],
			'model__accuracy':[2],
			'boolean__verbose':[True],
			'boolean__load':[True],
			'boolean__plot':[True],
			'plot_fig':[{'Loss':{},'BestFit':{},'Coef':{}}],
			'fit__kwargs__extrema__passes':[0],
			'fit__kwargs__loss_weights':[{'l2':1}],
			'fit__kwargs__score_weights':[{'l2':1}],
			'fit__kwargs__loss_scale':[1],
			'fit__kwargs__score_scale':[1],
			'fit__kwargs__estimator':['OLS'],
			'fit__kwargs__tol':[1e-15],
			'fit__kwargs__loss_func':['weighted'],
			'fit__kwargs__score_func':['weighted'],
		},
	}


	if len(sys.argv)>1:
		with open(sys.argv[1],'r') as fobj:
			_settings['sys'].update(json.load(fobj))

	for k in _settings:
		_settings[k].update({l: _defaults[k][l] for l in _defaults.get(k,{}) if l not in _settings.get(k,{})})


	_settings['main']['main__info'] = _settings['sys'].pop('main__info',['fit'])
	_settings['main']['main__dim'] = _settings['sys'].pop('main__dim',[2])	
	_settings['main']['main__keys'] = _settings['sys'].pop('main__keys',[
				{
				# 'vol':{'type':'standard','model':'vol'},
				# 'psi':{'type':'standard','model':'psi'},
				# 'vol':{'type':'symmetrized','model':'vol'},
				# 'psi':{'type':'symmetrized','model':'psi'},
				'vol':{'type':'physics','model':'vol'},
				'psi':{'type':'physics','model':'psi'},		
				# 'vol':{'type':'physics_symmetrized','model':'vol'},
				# 'psi':{'type':'physics_symmetrized','model':'psi'},				
				}
				])

	_settings['main']['main__groups'] = _settings['sys'].pop('main__groups',[[]])


	for parameters in permute_settings(_settings['main'],_set_settings=False):

		_settings['key'].update({k:{} for k in parameters['main__keys']})

		for key in parameters['main__keys']:
			if CLUSTER:
				_settings['sys']['sys__directories__cwd__load'] = [os.path.abspath(os.path.expanduser(d)) for d in _settings['sys'].pop('sys__directories__cwd__load',['input'])]
			else:
				_settings['sys']['sys__directories__cwd__load'] = [os.path.abspath(os.path.expanduser(d)) for d in _settings['sys'].pop('sys__directories__cwd__load',['input'])]


			_settings['sys']['sys__directories__cwd__dump'] = [os.path.abspath(os.path.expanduser(d)) for d in _settings['sys'].pop('sys__directories__cwd__dump',['output'])]




			_settings['sys']['sys__directories__directories__load'] = _settings['sys'].pop('sys__directories__directories__load',[['data']])
			_settings['sys']['sys__directories__directories__dump'] = _settings['sys'].pop('sys__directories__directories__dump',[['data']])
			_settings['sys']['sys__files__files'] = _settings['sys'].pop('sys__files__files',[['data.csv']])
			_settings['sys']['structure__samples'] = _settings['sys'].pop('structure__samples',[None])
			# _settings['sys']['structure__samples'] = [np.arange(1000,1500)]#_settings['sys'].pop('structure__samples',[None])
			_settings['sys']['structure__filters__passes'] = _settings['sys'].pop('structure__filters__passes',[3])
			_settings['sys']['model__weights'] = _settings['sys'].pop('model__weights',[_settings['sys'].get('model__weight',['stencil'])])
			_settings['sys']['model__adjacencies'] = _settings['sys'].pop('model__adjacencies',[_settings['sys'].get('model__adjacency',['nearest'])])
			_settings['sys']['model__accuracies'] = _settings['sys'].pop('model__accuracies',[_settings['sys'].get('model__accuracy',[2])])
			_settings['sys']['boolean__verbose'] = _settings['sys'].pop('boolean__verbose',[True])
			_settings['sys']['boolean__load'] = _settings['sys'].pop('boolean__load',[1])
			_settings['sys']['boolean__plot'] = _settings['sys'].pop('boolean__plot',[1])

			_settings['sys']['plot__fig'] = _settings['sys'].pop('plot__fig',[{'Loss':{},'BestFit':{},'Coef':{}}])
			
			_settings['sys']['fit__kwargs__extrema__passes'] = _settings['sys'].pop('fit__kwargs__extrema__passes',[0])
			_settings['sys']['fit__kwargs__loss_weights'] = _settings['sys'].pop('fit__kwargs__loss_weights',[{'l2':1}])
			_settings['sys']['fit__kwargs__score_weights'] = _settings['sys'].pop('fit__kwargs__score_weights',[{'l2':1}])
			_settings['sys']['fit__kwargs__loss_scale'] = _settings['sys'].pop('fit__kwargs__loss_scale',[1])
			_settings['sys']['fit__kwargs__score_scale'] = _settings['sys'].pop('fit__kwargs__score_scale',[1])
			_settings['sys']['fit__kwargs__estimator'] = _settings['sys'].pop('fit__kwargs__estimator',['OLS'])
			_settings['sys']['fit__kwargs__tol'] = _settings['sys'].pop('fit__kwargs__tol',[1e-15])
			_settings['sys']['fit__kwargs__loss_func'] = _settings['sys'].pop('fit__kwargs__loss_func',['weighted'])
			_settings['sys']['fit__kwargs__score_func'] = _settings['sys'].pop('fit__kwargs__score_func',['weighted'])
			
			for k,v in {
					'boolean':True,
					'sys':True,
					'structure':True,
					'terms':True,
					'model':True,
					'fit':True,
					'analysis':True,
					'plot':True,
				}.items():
				_settings['sys']['%s__verbose'%(k)] = [v]

			variables = ['mechanical','chemical','energy','time']
			components = ['label','dimension','other']
			types = ['standard','symmetrized','physics','physics_symmetrized']
			variables = {k: {s:{t:{} for t in types} for s in components}for k in variables}
			typesets = {'mechanical':r'\mathbf','chemical':'','energy':'','time':''}

			var = 'mechanical'
			variables[var]['label']['standard'] = {'E':r'\bar{E}'}
			variables[var]['label']['symmetrized'] = {'e':r'\bar{e}'}
			variables[var]['label']['physics'] = {'E':r'\bar{E}'}
			variables[var]['label']['physics_symmetrized'] = {'e':r'\bar{e}'}
			variables[var]['dimension']['standard'] = {y: {'%d%d'%(i,j):'%d%d'%(i,j) for i in range(1,parameters['main__dim']+1) for j in range(1,parameters['main__dim']+1) if i<=j} for y in variables[var]['label']['standard']}
			variables[var]['dimension']['symmetrized'] = {y: {'%d'%(i):'%d'%(i) for i in [1,2,6]} for y in variables[var]['label']['symmetrized']}
			variables[var]['dimension']['physics'] = {y: {'%d%d'%(i,j):'%d%d'%(i,j) for i in range(1,parameters['main__dim']+1) for j in range(1,parameters['main__dim']+1) if i<=j} for y in variables[var]['label']['physics']}
			variables[var]['dimension']['physics_symmetrized'] = {y: {'%d'%(i):'%d'%(i) for i in [2]} for y in variables[var]['label']['physics_symmetrized']}
			variables[var]['other']['standard'] = {'':'',DELIMETER:''}
			variables[var]['other']['symmetrized'] = {'':'',DELIMETER:''}
			variables[var]['other']['physics'] = {'':'',DELIMETER:''}
			variables[var]['other']['physics_symmetrized'] = {'':'',DELIMETER:''}
			
			var = 'chemical'
			variables[var]['label']['standard'] = {'vol':r'\varphi','len':r'l','number':r'N'} 
			variables[var]['label']['symmetrized'] = {'vol':r'\varphi','len':r'l','number':r'N'} 
			variables[var]['label']['physics'] = {'vol':r'\bar{\varphi}','len':r'\bar{l}','number':r'\bar{N}'} 
			variables[var]['label']['physics_symmetrized'] = {'vol':r'\bar{\varphi}','len':r'\bar{l}','number':r'\bar{N}'} 	
			variables[var]['dimension']['standard'] = {y: {'square':r'\square','rectangle_p':r'\hrectangle','rectangle_m':r'\vrectangle'} for y in variables[var]['label']['standard']}
			variables[var]['dimension']['symmetrized'] = {y: {'square':r'\square'} for y in variables[var]['label']['symmetrized']}
			variables[var]['dimension']['physics'] = {y: {**{'vol':{'1':r''}},**{x: {'1':r'1','2':r'2'} for x in ['len','number']}}[y] for y in variables[var]['label']['physics']}
			variables[var]['dimension']['physics_symmetrized'] = {y: {**{'vol':{'1':r''}},**{x: {'1':r'1','2':r'2'} for x in ['len','number']}}[y] for y in variables[var]['label']['physics_symmetrized']}
			variables[var]['other']['standard'] = {'':'',DELIMETER:r'\alpha'}
			variables[var]['other']['symmetrized'] = {'':'',DELIMETER:r'\alpha'}
			variables[var]['other']['physics'] = {'':'',DELIMETER:r'\alpha'}
			variables[var]['other']['physics_symmetrized'] = {'':'',DELIMETER:r'\alpha'}	

			var = 'energy'
			variables[var]['label']['standard'] = {'Psi':r'\Psi'}
			variables[var]['label']['symmetrized'] = {'Psi':r'\Psi'}
			variables[var]['label']['physics'] = {'Psi':r'\Psi'}
			variables[var]['label']['physics_symmetrized'] = {'Psi':r'\Psi'}	
			variables[var]['dimension']['standard'] = {y: {'tot':'','me':r'\textrm{mech}','chem':r'\textrm{chem}','couple':r'\textrm{couple}'} for y in variables[var]['label']['standard']}
			variables[var]['dimension']['symmetrized'] = {y: {'tot':'','me':r'\textrm{mech}','chem':r'\textrm{chem}','couple':r'\textrm{couple}'} for y in variables[var]['label']['symmetrized']}
			variables[var]['dimension']['physics'] = {y: {'tot':'','me':r'\textrm{mech}','chem':r'\textrm{chem}','couple':r'\textrm{couple}'} for y in variables[var]['label']['physics']}
			variables[var]['dimension']['physics_symmetrized'] = {y: {'tot':'','me':r'\textrm{mech}','chem':r'\textrm{chem}','couple':r'\textrm{couple}'} for y in variables[var]['label']['physics_symmetrized']}
			variables[var]['other']['standard'] = {} #{'':'',DELIMETER:''}
			variables[var]['other']['symmetrized'] = {} #{'':'',DELIMETER:''}
			variables[var]['other']['physics'] = {} #{'':'',DELIMETER:''}
			variables[var]['other']['physics_symmetrized'] = {} #{'':'',DELIMETER:''}	


			var = 'time'
			variables[var]['label']['standard'] = {'time':'t'}
			variables[var]['label']['symmetrized'] = {'time':'t'}
			variables[var]['label']['physics'] = {'time':'t'}
			variables[var]['label']['physics_symmetrized'] = {'time':'t'}	
			variables[var]['dimension']['standard'] = {y: {'':''} for y in variables[var]['label']['standard']}
			variables[var]['dimension']['symmetrized'] = {y: {'':''} for y in variables[var]['label']['symmetrized']}
			variables[var]['dimension']['physics'] = {y: {'':''} for y in variables[var]['label']['physics']}
			variables[var]['dimension']['physics_symmetrized'] = {y: {'':''} for y in variables[var]['label']['physics_symmetrized']}
			variables[var]['other']['standard'] = {'':'',DELIMETER:''}
			variables[var]['other']['symmetrized'] = {'':'',DELIMETER:''}	
			variables[var]['other']['physics'] = {'':'',DELIMETER:''}
			variables[var]['other']['physics_symmetrized'] = {'':'',DELIMETER:''}		



			typed = parameters['main__keys'][key]['type']
			samples = list(sampler(key,typed,variables,weight=_settings['sys']['model__weight'])())

			texstrings = {}
			_texstrings = {}

			texstrings.update({									 			
				**{'%s%s%s'%(s,y,'%s%s'%('' if x in [''] else '_',x)):r'{%s}%s'%(Y if ((s not in [DELIMETER]) and (x not in [''])) else 
																r'%s{%s}%s'%(typesets[k],Y,'_{%s}'%(variables[k]['other'][t].get(s,s)) 
																	if variables[k]['other'][t].get(s,s) not in [''] else variables[k]['other'][t].get(s,s)),
																'_{%s}'%(X if s not in [DELIMETER] else variables[k]['other'][t].get(s,X)))
					for k in variables
					for t in [typed]
					for s,S in {'':'',DELIMETER:''}.items()
					for y,Y in variables[k]['label'][t].items()
					for x,X in {'':'',**variables[k]['dimension'][t][y]}.items()
				}})
			texstrings.update({		
				**{'tot_c_ave':r'c',
				   'frame': r'\textrm{Time}',
				   },
				**{
					**{os.path.join(path,folder):r'{\mathcal{D}}_{\textrm{%s}}'%(folder) 
						for folder in directory
						for directory in _settings['sys']['sys__directories__directories__load']
						for path in _settings['sys']['sys__directories__cwd__load']
					},
				}
			}
			)


			_texstrings.update(texstrings)
			_texstrings.update({
				s+DELIMETER.join([*['partial' for i in range(o)],str(o),y,DELIMETER.join(x),*[w for i in range(o)]]):r'\frac{{%s}^{%s} %s}{%s}'%(S,str(o) if o>1 else '',Y,
						r' '.join(['%s %s'%(S,U) for U in X]) if len(set(X))>1 else r'{%s} {%s}^{%s}'%(S,X[0],str(o) if o>1 else ''))
					for t in [typed]
					for s,S in {'':r'\delta',DELIMETER:r'\partial'}.items()
					for o in range(1,3+1)									 		
					for w in _settings['sys']['model__weights'][0]
					for y,Y in {k: texstrings[k] for k in [
								*['%s%s'%(y,'%s%s'%('' if x in [''] else '_',x)) for y in variables['chemical']['label'][t] for x in variables['chemical']['dimension'][t][y]],
								*['%s%s'%(y,'%s%s'%('' if x in [''] else '_',x)) for y in variables['energy']['label'][t] for x in variables['energy']['dimension'][t][y] if x in ['tot','me']]
								]}.items()
					for x,X in {k: tuple(texstrings[i] for i in k) for k in itertools.product([ 
															*['%s%s'%(y,'%s%s'%('' if x in [''] else '_',x)) for y in variables['chemical']['label'][t] for x in variables['chemical']['dimension'][t][y]],
															*['%s%s'%(y,'%s%s'%('' if x in [''] else '_',x)) for y in variables['mechanical']['label'][t] for x in variables['mechanical']['dimension'][t][y]],
															],repeat=o)}.items()
				}
				)


			_texstrings.update({
				s+DELIMETER.join([*['partial' for i in range(o)],str(o),y,DELIMETER.join(x),*[w for i in range(o)]]):r'\frac{{%s}^{%s} %s}{%s}'%(S,str(o) if o>1 else '',Y,
						r' '.join(['%s %s'%(S,U) for U in X]) if len(set(X))>1 else r'{%s} {%s}^{%s}'%(S,X[0],str(o) if o>1 else ''))
					for t in [typed]
					for s,S in {'':r'\delta',DELIMETER:r'\partial'}.items()
					for o in range(1,1+1)									 		
					for w in ['stencil']
					for y,Y in {k: texstrings[k] for k in [
								*['%s%s'%(y,'%s%s'%('' if x in [''] else '_',x)) for y in variables['chemical']['label'][t] for x in variables['chemical']['dimension'][t][y]],
								]}.items()
					for x,X in {k: tuple(texstrings[i] for i in k) for k in itertools.product([ 
															*['%s'%(y) for y in variables['time']['label'][t]],
															],repeat=o)}.items()
															
				})


				# values = {
				# 	delimiter.join([
				# 	delimiter.join([o]*j),
				# 	str(j),
				# 	y,
				# 	delimiter.join([u for u in x]),
				# 	delimiter.join([w]*j)
				# 	]):func%(Symbol,str(j) if j>1 else '','%s'%(outputs[y]),
				# 			' '.join([r'%s %s'%(Symbol,inputs[v]) for v in x]) if len(set([v for v in x]))>1 else r'{%s %s}^{%s}'%(Symbol,inputs[[v for v in x][0]],str(j) if j>1 else ''),
				# 	)if usetex else (
				# 		'd^%s%s/%s'%(str(j) if j>1 else '',y,''.join(['d%s'%(v) for v in x]))
				# 		)				
				# 	for o in [o for o in args['operators'] if o in [symbol]]
				# 	for j in range(1,args['order']+1)
				# 	for y in outputs
				# 	for x in icombinations(inputs,[j],unique=args['unique'])
				# 	for w in args['weights']
				# 	}









			if key in ['vol']:
				_settings['key'][key]['model__weights'] = [weight for weight in _settings['sys']['model__weights']]
				_settings['key'][key]['model__adjacencies'] = [adj for adj in _settings['sys']['model__adjacencies']]
				_settings['key'][key]['model__accuracies'] = [acc for acc in _settings['sys']['model__accuracies']]
				_settings['key'][key]['model__order'] = [order for order in _settings['sys']['model__order']]
				_settings['key'][key]['model__order'] = [list(range(2,max(2,order)+1)) for order in _settings['key'][key]['model__order']]
				_settings['key'][key]['model__operations'] = [['partial'] for order in _settings['key'][key]['model__order']]
				_settings['key'][key]['model__inputs'] = [[
													*['%s%s'%(y,'%s%s'%('' if x in [''] else '_',x)) 
														for y in variables['mechanical']['label'][typed] 
														for x in variables['mechanical']['dimension'][typed][y]],	
													*['%s%s'%(y,'%s%s'%('' if x in [''] else '_',x)) 
														for y in variables['chemical']['label'][typed] 
														for x in variables['chemical']['dimension'][typed][y]
														],	
													*[DELIMETER.join([*['partial' for i in range(o)],str(o),y,DELIMETER.join(x),*[weight for i in range(o)]])
														for o in order									 		
														for y in ['%s%s'%(y,'%s%s'%('' if x in [''] else '_',x)) 
																for y in variables['energy']['label'][typed] 
																for x in variables['energy']['dimension'][typed][y]
																if x in ['tot']
																]
														for x in list(itertools.product(
																[*['%s%s'%(y,'%s%s'%('' if x in [''] else '_',x)) 
																for y in variables['chemical']['label'][typed] 
																for x in variables['chemical']['dimension'][typed][y]
																if y in ['vol']],
																*['%s%s'%(y,'%s%s'%('' if x in [''] else '_',x)) 
																for y in variables['mechanical']['label'][typed] 
																for x in variables['mechanical']['dimension'][typed][y]]
																],repeat=o))]
														] for order,weights in zip(_settings['key'][key]['model__order'],_settings['key'][key]['model__weights']) for weight in weights]
				_settings['key'][key]['model__outputs'] = [[DELIMETER.join([*['partial' for i in range(o)],str(o),y,DELIMETER.join(x),*[w for i in range(o)]])
														for w in ['stencil']
														for o in range(1,1+1)									 		
														for y in ['%s%s'%(y,'%s%s'%('' if x in [''] else '_',x)) 
																for y in variables['chemical']['label'][typed] 
																for x in variables['chemical']['dimension'][typed][y]
																if y in ['vol']
																]
														for x in list(itertools.product(
																[*['%s%s'%(y,'%s%s'%('' if x in [''] else '_',x)) 
																for y in variables['time']['label'][typed]
																for x in variables['time']['dimension'][typed][y]]															
																],repeat=o))
														]for order in _settings['key'][key]['model__order']]
				_settings['key'][key]['model__constants'] = [{w:[] for w in y} for x,y in zip(_settings['key'][key]['model__inputs'],_settings['key'][key]['model__outputs'])]
				_settings['key'][key]['model__manifold'] = [[
														*['%s%s'%(y,'%s%s'%('' if x in [''] else '_',x)) 
															for y in variables['mechanical']['label'][typed] 
															for x in variables['mechanical']['dimension'][typed][y]],	
														*['%s%s'%(y,'%s%s'%('' if x in [''] else '_',x)) 
															for y in variables['chemical']['label'][typed] 
															for x in variables['chemical']['dimension'][typed][y]
															if y in ['vol']] ] for inputs,outputs in zip(_settings['key'][key]['model__inputs'],_settings['key'][key]['model__outputs'])]
				_settings['key'][key]['model__p'] = [len([
													*['%s%s'%(y,'%s%s'%('' if x in [''] else '_',x)) 
														for y in variables['mechanical']['label'][typed] 
														for x in variables['mechanical']['dimension'][typed][y]],	
													*['%s%s'%(y,'%s%s'%('' if x in [''] else '_',x)) 
														for y in variables['chemical']['label'][typed] 
														for x in variables['chemical']['dimension'][typed][y]
														if y in ['vol']] ]) for inputs,outputs in zip(_settings['key'][key]['model__inputs'],_settings['key'][key]['model__outputs'])]

				_settings['key'][key]['model__size'] = [len(samples) for inputs in _settings['key'][key]['model__inputs']]
				_settings['key'][key]['terms__terms'] = [[
							*[{'function':y,'variable': list(x),
							'weight':['stencil']*(max(j,1)),
							'adjacency':['backward_nearest']*(max(j,1)),
							'dimension':[0]*max(j,0),
							'accuracy':[1]*max(j,1),							
							'manifold':[[
										*['%s%s'%(y,'%s%s'%('' if x in [''] else '_',x)) 
										for y in variables['time']['label'][typed]
										for x in variables['time']['dimension'][typed][y]]
										]]*(max(j,1)),				
							'order':j,
							'operation':[o]*(max(j,1)),
						   }
						   for o in operations
						   for j in [1] 
						   for y in ['%s%s'%(y,'%s%s'%('' if x in [''] else '_',x)) 
									for y in variables['chemical']['label'][typed] 
									for x in variables['chemical']['dimension'][typed][y]
									if y in ['vol']
									]
						   for x in [v for v in itertools.product([
										*['%s'%(y) 
										for y in variables['time']['label'][typed] ]
										],repeat=j)]],		
						  *[{'function':y,'variable': list(x),
							'weight':[weight[0]]*(max(j,1)),
							'adjacency':[adj[0]]*(max(j,1)),
							'manifold':[[
										*['%s%s'%(y,'%s%s'%('' if x in [''] else '_',x)) 
										for y in variables['chemical']['label'][typed] 
										for x in variables['chemical']['dimension'][typed][y]
										if y in ['vol']],
										*['%s%s'%(y,'%s%s'%('' if x in [''] else '_',x)) 
										for y in variables['mechanical']['label'][typed] 
										for x in variables['mechanical']['dimension'][typed][y]]
										]]*(max(j,1)),				
							'dimension': [[
										*['%s%s'%(y,'%s%s'%('' if x in [''] else '_',x)) 
										for y in variables['chemical']['label'][typed] 
										for x in variables['chemical']['dimension'][typed][y]
										if y in ['vol']],
										*['%s%s'%(y,'%s%s'%('' if x in [''] else '_',x)) 
										for y in variables['mechanical']['label'][typed] 
										for x in variables['mechanical']['dimension'][typed][y]]
										].index(u) for u in x],							
							'accuracy':[acc[0]]*max(j,1),
							'order':j,
							'operation':[o]*(max(j,1)),
						   }
						   for o in operations
						   for j in order 
						   for y in [
								*['%s%s'%(y,'%s%s'%('' if x in [''] else '_',x)) 
									for y in variables['energy']['label'][typed] 
									for x in variables['energy']['dimension'][typed][y]
									if x in ['tot']					
									]]								  
						   for x in [v for v in itertools.product([
										*['%s%s'%(y,'%s%s'%('' if x in [''] else '_',x)) 
										for y in variables['chemical']['label'][typed] 
										for x in variables['chemical']['dimension'][typed][y]
										if y in ['vol']],							
										*['%s%s'%(y,'%s%s'%('' if x in [''] else '_',x)) 
										for y in variables['mechanical']['label'][typed] 
										for x in variables['mechanical']['dimension'][typed][y]]							
										],repeat=j)]],
						   ] for operations,order,weight,adj,acc in zip(_settings['key'][key]['model__operations'],_settings['key'][key]['model__order'],
						   												_settings['key'][key]['model__weights'],_settings['key'][key]['model__adjacencies'],
						   												_settings['key'][key]['model__accuracies'])]


				_settings['key'][key]['model__rhs_lhs'] = [{DELIMETER.join([y,*inputs,str(i)]):{'rhs':[inputs],'lhs':[y]}
															for y in outputs[:]}
															for i,(inputs,outputs) in enumerate(zip(_settings['key'][key]['model__inputs'],_settings['key'][key]['model__outputs']))]
				_settings['key'][key]['model__basis'] =['polynomial' for inputs,outputs in zip(_settings['key'][key]['model__inputs'],_settings['key'][key]['model__outputs'])]
				_settings['key'][key]['model__selection'] =[samples for inputs in _settings['key'][key]['model__inputs']]
				_settings['key'][key]['model__format'] =['csr' for inputs in _settings['key'][key]['model__inputs']]
				_settings['key'][key]['model__sparsity'] =[100 for inputs in _settings['key'][key]['model__inputs']]
				_settings['key'][key]['model__unique'] =[True for inputs in _settings['key'][key]['model__inputs']]
				_settings['key'][key]['model__chunk'] =[1 for inputs in _settings['key'][key]['model__inputs']]

				_settings['key'][key]['model__intercept_'] =[True for inputs in _settings['key'][key]['model__inputs']]
				_settings['key'][key]['model__type'] = ['ode' for inputs,outputs in zip(_settings['key'][key]['model__inputs'],_settings['key'][key]['model__outputs'])]

				_settings['key'][key]['texify__texargs__inputs'] =	[{k:_texstrings[k] for k in inputs} for inputs in _settings['key'][key]['model__inputs']]										
				_settings['key'][key]['texify__texargs__outputs'] =	[{k:_texstrings[k] for k in outputs} for outputs in _settings['key'][key]['model__outputs']]									  
				

				_settings['key'][key]['texify__texargs__groups'] = [{'%s%s%s'%(s,y,'%s%s'%('' if x in [''] else '_',x)):r'{%s}'%(r'%s{%s}'%(typesets[k],Y) if ((s not in [DELIMETER]) and (x not in [''])) else 
																					r'%s{%s}'%(typesets[k],Y))
																		for k in variables
																		for t in [typed]
																		for s,S in {'':''}.items()
																		for y,Y in variables[k]['label'][t].items()
																		for x,X in {'':'',**variables[k]['dimension'][t][y]}.items()
																	if '%s%s%s'%(s,y,'%s%s'%('' if x in [''] else '_',x)) in inputs
																	} for inputs,outputs in zip(_settings['key'][key]['model__inputs'],_settings['key'][key]['model__outputs'])]

				_settings['key'][key]['texify__texstrings'] = [_texstrings for inputs,outputs in zip(_settings['key'][key]['model__inputs'],_settings['key'][key]['model__outputs'])]
				


				_settings['key'][key]['texify__texargs__bases'] = [{'monomial':1,'polynomial':0,'taylorseries':0,'derivative':0,'expansion':0,k:1} for k in _settings['key'][key]['model__basis']]
				_settings['key'][key]['texify__texargs__selection'] =  [samples for inputs in _settings['key'][key]['model__inputs']]
				_settings['key'][key]['plot__settings__BestFit__other__y'] = [outputs[:1] for inputs,outputs in zip(_settings['key'][key]['model__inputs'],_settings['key'][key]['model__outputs'])]
				# _settings['key'][key]['plot__settings__BestFit__other__iterations'] = [[None,30,20,10,5] for inputs,outputs in zip(_settings['key'][key]['model__inputs'],_settings['key'][key]['model__outputs'])]
				_settings['key'][key]['plot__settings__Loss__other__y'] = [outputs[:1] for inputs,outputs in zip(_settings['key'][key]['model__inputs'],_settings['key'][key]['model__outputs'])]
				_settings['key'][key]['plot__settings__Coef__other__y'] = [outputs[:1] for inputs,outputs in zip(_settings['key'][key]['model__inputs'],_settings['key'][key]['model__outputs'])]
				
				_settings['key'][key]['plot__settings__Variables__other__y'] = [[
																*[['%s%s'%(y,'%s%s'%('' if x in [''] else '_',x)) 
																	for y in variables['energy']['label'][typed]
																	if x in variables['energy']['dimension'][typed][y] 
																	]
																	for x in ['tot','me']
																	],					
																*[['%s%s'%(y,'%s%s'%('' if x in [''] else '_',x)) 
																	for y in variables['mechanical']['label'][typed] 
																	for x in variables['mechanical']['dimension'][typed][y]]],
																*[['%s%s'%(y,'%s%s'%('' if x in [''] else '_',x)) 
																	for x in variables['chemical']['dimension'][typed][y]]
																	for y in variables['chemical']['label'][typed]],
																]for inputs,outputs in zip(_settings['key'][key]['model__inputs'],_settings['key'][key]['model__outputs'])]
				_settings['key'][key]['plot__settings__BestFit__other__x'] = [['frame'] for inputs,outputs in zip(_settings['key'][key]['model__inputs'],_settings['key'][key]['model__outputs'])]
				_settings['key'][key]['plot__settings__Loss__other__x'] = [['frame'] for inputs,outputs in zip(_settings['key'][key]['model__inputs'],_settings['key'][key]['model__outputs'])]
				_settings['key'][key]['plot__settings__Coef__other__x'] = [['frame'] for inputs,outputs in zip(_settings['key'][key]['model__inputs'],_settings['key'][key]['model__outputs'])]
				_settings['key'][key]['plot__settings__Variables__other__x'] = [[['frame']*len(_y) for _y in y] for y in _settings['key'][key]['plot__settings__Variables__other__y']]

				_settings['key'][key]['plot__settings__Loss__ax__set_ylim'] = [{
																			'fit':{'ymin':1e-5,'ymax':1e-1},
																			'approach':{'ymin':1e-2,'ymax':1e-1},
																			'predict':{'ymin':1e-4,'ymax':1e3},
																			'kmeans':{'ymin':1e-4,'ymax':1e3}}[parameters['main__info']] for inputs,outputs in zip(_settings['key'][key]['model__inputs'],_settings['key'][key]['model__outputs'])]
				_settings['key'][key]['plot__settings__Variables__style__layout__nrows'] = [2 for inputs,outputs in zip(_settings['key'][key]['model__inputs'],_settings['key'][key]['model__outputs'])]
				_settings['key'][key]['plot__settings__Variables__style__layout__ncols'] = [int(len(y)/nrows)+(len(y)%nrows) for y,nrows in zip(_settings['key'][key]['plot__settings__Variables__other__y'],
																																	  _settings['key'][key]['plot__settings__Variables__style__layout__nrows'])]
				_settings['key'][key]['plot__settings__Loss__ax__plot__label'] = ['%s' for inputs,outputs,weight in zip(_settings['key'][key]['model__inputs'],_settings['key'][key]['model__outputs'],_settings['key'][key]['model__weights'])]
				_settings['key'][key]['plot__settings__Loss__ax__legend__loc'] = ['lower right' for inputs,outputs,weight in zip(_settings['key'][key]['model__inputs'],_settings['key'][key]['model__outputs'],_settings['key'][key]['model__weights'])]
				_settings['key'][key]['plot__settings__BestFit__ax__legend__loc'] = ['lower right' for inputs,outputs,weight in zip(_settings['key'][key]['model__inputs'],_settings['key'][key]['model__outputs'],_settings['key'][key]['model__weights'])]
				_settings['key'][key]['plot__settings__Loss__other__constant'] = [{'fit':['marker'],'predict':['marker'],'approach':['marker'],'kmeans':[]}[parameters['main__info']] for inputs,outputs,weight in zip(_settings['key'][key]['model__inputs'],_settings['key'][key]['model__outputs'],_settings['key'][key]['model__weights'])]


				_settings['key'][key]['plot__settings__Operators__other__terms'] = [ [{'function':['Psi_tot'],'variable':['vol_square'],'order':[1]},
																				  {'function':['Psi_tot'],'variable':['vol_square','E11'],'order':[2]}] 
																				  for inputs,outputs,weight in zip(_settings['key'][key]['model__inputs'],
																													_settings['key'][key]['model__outputs'],
																													_settings['key'][key]['model__weights'])]


				_settings['key'][key]['structure__rename'] = [{
								**{
									'%s%s'%(y,'%s%s'%('' if x in [''] else '_',x)):'%s_%s'%(y,X) 
									for y in ['vol','len','number']
									for x,X in {'%s_%s'%('square',x if x not in ['square'] else 'rectangle'):x for x in ['square','rectangle_p','rectangle_m']}.items()
									},
								**{'%s%s'%(y,x):'%s%s'%(y,'%s%s'%('' if x in [''] else '_',x)) 
										for y in ['E']
										for x in ['%d%d'%(i,j) for i in range(1,parameters['main__dim']+1) for j in range(1,parameters['main__dim']+1)]},
								**{'Psi':'Psi_tot'}
								}
								for inputs,outputs in zip(_settings['key'][key]['model__inputs'],_settings['key'][key]['model__outputs'])]					

				_settings['key'][key]['structure__scale'] = [{'labels':[*inputs,*outputs]} for inputs,outputs in zip(_settings['key'][key]['model__inputs'],_settings['key'][key]['model__outputs'])]
				_settings['key'][key]['structure__functions'] = [[
															{'labels':'e_1',
															 'func': {
																2: (lambda df: (2/np.sqrt(2))*((df['E_11']+df['E_22']).values)),
																3: (lambda df: (1/np.sqrt(3))*((df['E_11']+df['E_22']+df['E_33']).values)),
																}[parameters['main__dim']],
															},
															{'labels':'e_2',
															 'func': {
																2:(lambda df: (2/np.sqrt(2))*((df['E_11']-df['E_22']).values)),
																3:(lambda df: (1/np.sqrt(2))*((df['E_11']-df['E_22']).values)),
																}[parameters['main__dim']],
															},
															{'labels':'e_6',
															 'func': {
																2:(lambda df: (2/np.sqrt(2))*((df['E_12']).values)),
																3:(lambda df: (2/np.sqrt(2))*((df['E_12']).values)),
																}[parameters['main__dim']],																														
															},
															{'labels':'len_1',
															 'func': lambda df: (df['len_square']).values,
															},
															{'labels':'len_2',
															 'func': lambda df: (1/2)*(df['len_rectangle_p']+df['len_rectangle_m']-df['len_square']).values,
															},
															{'labels':'vol_1',
															 'func': lambda df: (df['vol_square']).values,
															},
															{'labels':'vol_2',
															 'func': lambda df: (1)*(df['vol_rectangle_p']+df['vol_rectangle_m']).values,
															},
															{'labels':'number_1',
															 'func': lambda df: (df['number_square']).values,
															},
															{'labels':'number_2',
															 'func': lambda df: (1)*(df['number_rectangle_p']+df['number_rectangle_m']).values,
															},
														 ] for inputs,outputs in zip(_settings['key'][key]['model__inputs'],_settings['key'][key]['model__outputs'])]
				_settings['key'][key]['structure__filters'] = [

					[
						{
						  'type':'rolling',
						  'labels':[
								*['%s%s'%(y,'%s%s'%('' if x in [''] else '_',x)) 
								for y in variables['mechanical']['label'][typed] 
								for x in variables['mechanical']['dimension'][typed][y]],	
								*['%s%s'%(y,'%s%s'%('' if x in [''] else '_',x)) 
								for y in variables['chemical']['label'][typed] 
								for x in variables['chemical']['dimension'][typed][y]
								if y not in ['number']],
								*['%s%s'%(y,'%s%s'%('' if x in [''] else '_',x)) 
								for y in variables['energy']['label'][typed] 
								for x in variables['energy']['dimension'][typed][y]],									
								],
						  'passes':i,
						  'rolling_kwargs':lambda df,n:{
									'win_type':'boxcar',
									'axis':0,
									'window':int(df.shape[0]/50) if df.shape[0] > 1000 else 25,
									'min_periods':1,
									'center':True,
									'closed':'both',
									},
							'mean_kwargs':{},

						  }
						  ] 
					for i in _settings['sys']['structure__filters__passes']
					for inputs,outputs in zip(_settings['key'][key]['model__inputs'],_settings['key'][key]['model__outputs'])
				]


			elif key in ['psi']:
				_settings['key'][key]['model__weights'] = [weight for weight in _settings['sys']['model__weights']]
				_settings['key'][key]['model__order'] = [order for order in _settings['sys']['model__order']]
				_settings['key'][key]['model__adjacencies'] = [adj for adj in _settings['sys']['model__adjacencies']]
				_settings['key'][key]['model__accuracies'] = [acc for acc in _settings['sys']['model__accuracies']]				
				_settings['key'][key]['model__operations'] = [['partial'] for order in _settings['key'][key]['model__order']]
				_settings['key'][key]['model__inputs'] = [[
													*['%s%s'%(y,'%s%s'%('' if x in [''] else '_',x)) 
														for y in variables['chemical']['label'][typed] 
														for x in variables['chemical']['dimension'][typed][y]
														if y in ['vol']
														],				
													*['%s%s'%(y,'%s%s'%('' if x in [''] else '_',x)) 
														for y in variables['mechanical']['label'][typed] 
														for x in variables['mechanical']['dimension'][typed][y]],		
													] for order,weight in zip(_settings['key'][key]['model__order'],_settings['key'][key]['model__weights'])]

				_settings['key'][key]['model__outputs'] = [[*['%s%s'%(y,'%s%s'%('' if x in [''] else '_',x)) 
														for y in variables['energy']['label'][typed] 
														for x in variables['energy']['dimension'][typed][y]
														if x in ['tot']
														],	
														] for order,weight in zip(_settings['key'][key]['model__order'],_settings['key'][key]['model__weights'])]
				_settings['key'][key]['model__constants'] = [{y:[] for y in outputs} for inputs,outputs in zip(_settings['key'][key]['model__inputs'],_settings['key'][key]['model__outputs'])]
				_settings['key'][key]['model__p'] = [len(inputs) for inputs,outputs in zip(_settings['key'][key]['model__inputs'],_settings['key'][key]['model__outputs'])]

				_settings['key'][key]['model__manifold'] = [[*inputs] for inputs,outputs in zip(_settings['key'][key]['model__inputs'],_settings['key'][key]['model__outputs'])]

				_settings['key'][key]['model__size'] = [None for inputs in _settings['key'][key]['model__inputs']]
				_settings['key'][key]['model__rhs_lhs'] = [None for inputs in _settings['key'][key]['model__inputs']]
				_settings['key'][key]['model__basis'] =['taylorseries' for inputs,outputs in zip(_settings['key'][key]['model__inputs'],_settings['key'][key]['model__outputs'])]
				_settings['key'][key]['model__selection'] =[sampler for inputs in _settings['key'][key]['model__inputs']]
				_settings['key'][key]['model__format'] =['csr' for inputs in _settings['key'][key]['model__inputs']]
				_settings['key'][key]['model__sparsity'] =[60 for inputs in _settings['key'][key]['model__inputs']]
				_settings['key'][key]['model__unique'] =[True for inputs in _settings['key'][key]['model__inputs']]
				_settings['key'][key]['model__chunk'] =[0.5 for inputs in _settings['key'][key]['model__inputs']]
				_settings['key'][key]['model__intercept_'] =[False for inputs in _settings['key'][key]['model__inputs']]

				_settings['key'][key]['texify__texargs__inputs'] =	[{k:_texstrings[k] for k in inputs} for inputs in _settings['key'][key]['model__inputs']]										
				_settings['key'][key]['texify__texargs__outputs'] =	[{k:_texstrings[k] for k in outputs} for outputs in _settings['key'][key]['model__outputs']]									  
				_settings['key'][key]['texify__texstrings'] = [texstrings for inputs,outputs in zip(_settings['key'][key]['model__inputs'],_settings['key'][key]['model__outputs'])]
				
				_settings['key'][key]['texify__texargs__groups'] = [{'%s%s%s'%(s,y,'%s%s'%('' if x in [''] else '_',x)):r'{%s}'%(r'%s{%s}'%(typesets[k],Y) if ((s not in [DELIMETER]) and (x not in [''])) else 
																					r'%s{%s}'%(typesets[k],Y))
																		for k in variables
																		for t in [typed]
																		for s,S in {'':''}.items()
																		for y,Y in variables[k]['label'][t].items()
																		for x,X in {'':'',**variables[k]['dimension'][t][y]}.items()
																	if '%s%s%s'%(s,y,'%s%s'%('' if x in [''] else '_',x)) in inputs
																	} for inputs,outputs in zip(_settings['key'][key]['model__inputs'],_settings['key'][key]['model__outputs'])]

				_settings['key'][key]['texify__texargs__iter'] = [{'monomial':0,'polynomial':0,'taylorseries':1,k:1} for k in _settings['key'][key]['model__basis']]
				_settings['key'][key]['texify__texargs__selection'] =  [slice(None) for inputs in _settings['key'][key]['model__inputs']]

				_settings['key'][key]['plot__settings__BestFit__other__y'] = [outputs[:1] for inputs,outputs in zip(_settings['key'][key]['model__inputs'],_settings['key'][key]['model__outputs'])]
				# _settings['key'][key]['plot__settings__BestFit__other__iterations'] = [[None,30,20,10,5] for inputs,outputs in zip(_settings['key'][key]['model__inputs'],_settings['key'][key]['model__outputs'])]
				_settings['key'][key]['plot__settings__Loss__other__y'] = [outputs[:1] for inputs,outputs in zip(_settings['key'][key]['model__inputs'],_settings['key'][key]['model__outputs'])]
				_settings['key'][key]['plot__settings__Coef__other__y'] = [outputs[:1] for inputs,outputs in zip(_settings['key'][key]['model__inputs'],_settings['key'][key]['model__outputs'])]
				_settings['key'][key]['plot__settings__Variables__other__y'] = [[
																*[['%s%s'%(y,'%s%s'%('' if x in [''] else '_',x)) 
																	for y in variables['energy']['label'][typed]
																	if x in variables['energy']['dimension'][typed][y] 
																	]
																	for x in ['tot','me']
																	],					
																*[['%s%s'%(y,'%s%s'%('' if x in [''] else '_',x)) 
																	for y in variables['mechanical']['label'][typed] 
																	for x in variables['mechanical']['dimension'][typed][y]]],
																*[['%s%s'%(y,'%s%s'%('' if x in [''] else '_',x)) 
																	for x in variables['chemical']['dimension'][typed][y]]
																	for y in variables['chemical']['label'][typed]],
																]for inputs,outputs in zip(_settings['key'][key]['model__inputs'],_settings['key'][key]['model__outputs'])]
				_settings['key'][key]['plot__settings__BestFit__other__x'] = [['frame'] for inputs,outputs in zip(_settings['key'][key]['model__inputs'],_settings['key'][key]['model__outputs'])]
				_settings['key'][key]['plot__settings__Loss__other__x'] = [['frame'] for inputs,outputs in zip(_settings['key'][key]['model__inputs'],_settings['key'][key]['model__outputs'])]
				_settings['key'][key]['plot__settings__Coef__other__x'] = [['frame'] for inputs,outputs in zip(_settings['key'][key]['model__inputs'],_settings['key'][key]['model__outputs'])]
				_settings['key'][key]['plot__settings__Variables__other__x'] = [[['frame']*len(_y) for _y in y] for y in _settings['key'][key]['plot__settings__Variables__other__y']]

				_settings['key'][key]['plot__settings__Loss__ax__set_ylim'] = [{'fit':{'ymin':5e-3,'ymax':5e-1},
																			'predict':{'ymin':1e-4,'ymax':1e3},
																			'approach':{'ymin':1e-2,'ymax':1e-1},																		
																			'kmeans':{'ymin':1e-4,'ymax':1e3}}[parameters['main__info']] for inputs,outputs in zip(_settings['key'][key]['model__inputs'],_settings['key'][key]['model__outputs'])]
				_settings['key'][key]['plot__settings__Variables__style__layout__nrows'] = [2 for inputs,outputs in zip(_settings['key'][key]['model__inputs'],_settings['key'][key]['model__outputs'])]
				_settings['key'][key]['plot__settings__Variables__style__layout__ncols'] = [int(len(y)/nrows)+(len(y)%nrows) for y,nrows in zip(_settings['key'][key]['plot__settings__Variables__other__y'],
																																	  _settings['key'][key]['plot__settings__Variables__style__layout__nrows'])]
				_settings['key'][key]['plot__settings__Loss__ax__plot__label'] = ['%s' for inputs,outputs,weight in zip(_settings['key'][key]['model__inputs'],_settings['key'][key]['model__outputs'],_settings['key'][key]['model__weights'])]
				_settings['key'][key]['plot__settings__Loss__ax__legend__loc'] = ['lower right' for inputs,outputs,weight in zip(_settings['key'][key]['model__inputs'],_settings['key'][key]['model__outputs'],_settings['key'][key]['model__weights'])]
				_settings['key'][key]['plot__settings__Loss__other__constant'] = [{'fit':['marker'],'predict':['marker'],'approach':['marker'],'kmeans':[]}[parameters['main__info']] for inputs,outputs,weight in zip(_settings['key'][key]['model__inputs'],_settings['key'][key]['model__outputs'],_settings['key'][key]['model__weights'])]

				_settings['key'][key]['plot__settings__Operators__other__terms'] = [ [{'function':['Psi_tot'],'variable':['vol_square'],'order':[1]},
																				  {'function':['Psi_tot'],'variable':['vol_square','E11'],'order':[2]}] 
																				  for inputs,outputs,weight in zip(_settings['key'][key]['model__inputs'],
																													_settings['key'][key]['model__outputs'],
																													_settings['key'][key]['model__weights'])]

				_settings['key'][key]['texify__texargs__bases'] = [{'monomial':1,'polynomial':0,'taylorseries':1,'derivative':1,'expansion':1,k:1} for k in _settings['key'][key]['model__basis']]

				_settings['key'][key]['structure__rename'] = [{
								**{
									'%s%s'%(y,'%s%s'%('' if x in [''] else '_',x)):'%s_%s'%(y,X) 
									for y in ['vol','len','number']
									for x,X in {'%s_%s'%('square',x if x not in ['square'] else 'rectangle'):x for x in ['square','rectangle_p','rectangle_m']}.items()
									},
								**{'%s%s'%(y,x):'%s%s'%(y,'%s%s'%('' if x in [''] else '_',x)) 
										for y in ['E']
										for x in ['%d%d'%(i,j) for i in range(1,parameters['main__dim']+1) for j in range(1,parameters['main__dim']+1)]},
								**{'Psi':'Psi_tot'}
								}
								for inputs,outputs in zip(_settings['key'][key]['model__inputs'],_settings['key'][key]['model__outputs'])]				
				_settings['key'][key]['structure__scale'] = [{'labels':[*inputs,*outputs]} for inputs,outputs in zip(_settings['key'][key]['model__inputs'],_settings['key'][key]['model__outputs'])]
				_settings['key'][key]['structure__functions'] = [[
															{'labels':'e_1',
															 'func': {
																2: (lambda df: (2/np.sqrt(2))*((df['E_11']+df['E_22']).values)),
																3: (lambda df: (1/np.sqrt(3))*((df['E_11']+df['E_22']+df['E_33']).values)),
																}[parameters['main__dim']],
															},
															{'labels':'e_2',
															 'func': {
																2:(lambda df: (2/np.sqrt(2))*((df['E_11']-df['E_22']).values)),
																3:(lambda df: (1/np.sqrt(2))*((df['E_11']-df['E_22']).values)),
																}[parameters['main__dim']],
															},
															{'labels':'e_6',
															 'func': {
																2:(lambda df: (2/np.sqrt(2))*((df['E_12']).values)),
																3:(lambda df: (2/np.sqrt(2))*((df['E_12']).values)),
																}[parameters['main__dim']],																														
															},
															{'labels':'len_1',
															 'func': lambda df: (df['len_square']).values,
															},
															{'labels':'len_2',
															 'func': lambda df: (1/2)*(df['len_rectangle_p']+df['len_rectangle_m']-df['len_square']).values,
															},
															{'labels':'vol_1',
															 'func': lambda df: (df['vol_square']).values,
															},
															{'labels':'vol_2',
															 'func': lambda df: (1)*(df['vol_rectangle_p']+df['vol_rectangle_m']).values,
															},
															{'labels':'number_1',
															 'func': lambda df: (df['number_square']).values,
															},
															{'labels':'number_2',
															 'func': lambda df: (1)*(df['number_rectangle_p']+df['number_rectangle_m']).values,
															},
														 ] for inputs,outputs in zip(_settings['key'][key]['model__inputs'],_settings['key'][key]['model__outputs'])]
				_settings['key'][key]['structure__filters'] = [

				[
					{
					  'type':'rolling',
					  'labels':[
							*['%s%s'%(y,'%s%s'%('' if x in [''] else '_',x)) 
							for y in variables['mechanical']['label'][typed] 
							for x in variables['mechanical']['dimension'][typed][y]],	
							*['%s%s'%(y,'%s%s'%('' if x in [''] else '_',x)) 
							for y in variables['chemical']['label'][typed] 
							for x in variables['chemical']['dimension'][typed][y]
							if y not in ['number']],
							*['%s%s'%(y,'%s%s'%('' if x in [''] else '_',x)) 
							for y in variables['energy']['label'][typed] 
							for x in variables['energy']['dimension'][typed][y]],									
							],
					  'passes':i,
					  'rolling_kwargs':lambda df,n:{
								'win_type':'boxcar',
								'axis':0,
								'window':int(df.shape[0]/50) if df.shape[0] > 1000 else 15,
								'min_periods':1,
								'center':True,
								'closed':'both',
								},
						'mean_kwargs':{},

					  }] 
					for i in _settings['sys']['structure__filters__passes']
					for inputs,outputs in zip(_settings['key'][key]['model__inputs'],_settings['key'][key]['model__outputs'])
				]




			_settings['default'] = {
				'sys__label':['test_%s'%(parameters['main__info'])],
				'sys__labels':[['sys__label','fit__kwargs__loss_weights','fit__kwargs__extrema__passes','structure__parameter',
								'model__basis','model__order','model__weight','model__approach',]],					
				'model__approach':['indiv'],
				'fit__info':[{
					}],
						# **({
						# 	'predict':{
						# 		'predict':{os.path.join(c,d) if load else os.path.join(c,d):{
						# 			'type':'fit',
						# 			'key':'all',
						# 			'keys':[os.path.join(c,d) if load else os.path.join(c,d)],
						# 			}
						# 			for dirs in (_settings['sys']['sys__directories__directories__dump'] if load else _settings['sys']['sys__directories__directories__load'][:])
						# 			for d in dirs
						# 			for c in (_settings['sys']['sys__directories__cwd__dump'] if load else _settings['sys']['sys__directories__cwd__load'])} 
						# 		},
						# 	'fit':{
						# 		'fit':{os.path.join(c,d) if load else os.path.join(c,d):{
						# 			'type':'fit',
						# 			'key':os.path.join(c,d) if load else os.path.join(c,d),
						# 			'keys':[os.path.join(c,d) if load else os.path.join(c,d)],
						# 			}
						# 			for dirs in (_settings['sys']['sys__directories__directories__dump'] if load else _settings['sys']['sys__directories__directories__load'][:])
						# 			for d in dirs									
						# 			for c in (_settings['sys']['sys__directories__cwd__dump'] if load else _settings['sys']['sys__directories__cwd__load'])}
						# 		},
						# 	'approach':{
						# 		'approach':{os.path.join(c,d) if load else os.path.join(c,d):{
						# 			'type':'fit',
						# 			'key':'all',
						# 			'keys':[os.path.join(c,d) if load else os.path.join(c,d)],
						# 			}									
						# 			for dirs in (_settings['sys']['sys__directories__directories__dump'] if load else _settings['sys']['sys__directories__directories__load'][:])
						# 			for d in dirs									
						# 			for c in (_settings['sys']['sys__directories__cwd__dump'] if load else _settings['sys']['sys__directories__cwd__load'])} 
						# 		},														
						# 	'kmeans':{
						# 		'fit':{os.path.join(c,d) if load else os.path.join(c,d):{
						# 			'type':'fit',
						# 			'key':os.path.join(c,d) if load else os.path.join(c,d),
						# 			'keys': select,
						# 			} 
						# 			for dirs in (_settings['sys']['sys__directories__directories__dump'] if load else _settings['sys']['sys__directories__directories__load'][:])
						# 			for d in dirs									
						# 			for c in (_settings['sys']['sys__directories__cwd__dump'] if load else _settings['sys']['sys__directories__cwd__load'])}
						# 		},
						# 	}[parameters['main__info']]) 
						#    }
						# 	for load in _settings['sys']['boolean__load']
						# 	],


				# 'plot__fig': [{'Loss':{},'BestFit':{},'Coef':{},'Variables':{}}],
				# 'plot__fig': [{'Loss':{},'BestFit':{},'Coef':{},'Variables':{}}],
				# 'plot__fig': [{'BestFit':{},}],
				# 'plot__fig': [{'Variables':{}}],
				# 'plot__axes': [{'Loss':{},'BestFit':{},'Coef':{},'Variables':{},'Error':{}}],
				'plot__axes': [{'Loss':{},'BestFit':{},'Coef':{}}],
				# 'plot__fig': [{'Loss':{}}],
				# 'plot__fig': [{'BestFit':None,'Coef':None}],
				# 'plot__fig': [{'BestFit':None,'Coef':None}],
				# 'plot__axes': [{'BestFit':None,'Coef':None}],
				# 'plot__fig': [{'Loss':None}],
				# 'plot__axes': [{'Loss':None}],
				# 'plot__fig': [{'BestFit':None}],
				# 'plot__axes': [{'BestFit':None}],
				# 'plot__fig': [{'Coef':None}],
				# 'plot__axes': [{'Coef':None}],
				# 'plot__fig': [{'Variables':None}],
				# 'plot__axes': [{'Variables':None}],				
				'plot__settings__BestFit__other__iterations': [ i for plot in _settings['sys']['boolean__plot'] 
																for i in ([[None],[30],[20],[10],[5]] if (0*plot) else [[None,30,20,10,5]])],
				'plot__retain__label':[{'Loss':1,'BestFit':0,'Coef':0,'Variables':0}],
				'plot__retain__dim':[{'Loss':1,'BestFit':0,'Coef':0,'Variables':0}],
				'plot__retain__key':[{'Loss':1,'BestFit':0,'Coef':0,'Variables':0}],
				'texify__usetex':[1],
			}



			_settings['all'] = {
						**_settings['default'],
						**_settings['key'][key],
						**_settings['sys'],
						}
			_groups = [[k for k in _settings['key'][key]],*parameters['main__groups']]

			_settings['grid'] = permute_settings(_settings['all'],_copy=False,_groups=_groups)


			for settings in _settings['grid']:

				data = {}
				metadata = {}
				main(data,metadata,settings)		
