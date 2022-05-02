#!/usr/bin/env python

# Import python modules
import sys,os,glob,copy,itertools
from natsort import natsorted
import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors 

# Global Variables
DELIMITER='__'

# Import user modules
from utility.texify import Texify,scinotation
from utility.plot import plot
from utility.dictionary import _set,_get,_pop,_has,_update,_permute,_formatstring
from utility.load_dump import load,dump,path_split,path_join
		

# Logging
import logging,logging.handlers
log = 'info'
logger = logging.getLogger(__name__)
#logger.setLevel(getattr(logging,log.upper()))




def _replace(string,replacements):
	for r in replacements:
		string = string.replace(r,replacements[r])
	return string


def _get_plot(x,y,key,typed,label_dim,name,metadata,constant=[]):
	import matplotlib
	from matplotlib import pyplot as plt
	params = {}
	if name in ['Loss']:
		stats = metadata[key][typed][label_dim]['stats']

		fields = {'stats':{'keys':['color'],'key':['marker','markersize','zorder']}}
		for method in fields:
			for field in fields[method]:
				classes = natsorted(list(set([tuple(metadata[k][typed][l][method][field]) if isinstance(metadata[k][typed][l][method][field],(list,tuple)) else metadata[k][typed][l][method][field]
									for k in metadata 									
									for l in (metadata[k][typed] if typed in metadata[k] else [])
									if typed in metadata[k] and method in metadata[k][typed][l]
									])))
				n_classes = len(classes)
				try:
					i_classes = classes.index(tuple(stats[field]) if isinstance(stats[field],(list,tuple)) else stats[field])
				except ValueError:
					try:
						i_classes = [i for i,c in enumerate(classes) if stats[field] in c][0]
					except:
						i_classes = 0
				for param in fields[method][field]:
					if param in constant:
						continue
					if method in ['stats'] and field in ['keys','key'] and param in ['color']:
						# values = getattr(plt.cm,'tab10')(np.linspace(0,1,n_classes))
						values = getattr(plt.cm,'tab10') #(np.linspace(0,1,n_classes))
						# values = getattr(plt.cm,'viridis')(np.linspace(0,1,n_classes))

						# values = {
							# 'fit':[values(i%n_classes) for i,c in enumerate(classes) if stats[field] in c],
							# 'predict':[None],
							# 'fit':[values(i%n_classes) for i,c in enumerate(classes) if stats[field] in c],							
							# }.get(typed,values)

						values = {
							'fit':['k'],
							'predict':[None],
							'interpolate':[None],
								  }.get(typed,values)								  
						value = values[i_classes%len(values)]						
						params[param] = value
					elif method in ['stats'] and field in ['keys','key'] and param in ['marker']:
						values = ['o','s','*','^','d','<','>','+','x','8']
						values = {'fit':values,'predict':['s',*values[1:]],'interpolate':values}.get(typed,values)
						value = values[i_classes%len(values)]
						# value = np.random.permutation(values)[i_classes%len(values)]
						params[param] = value
					elif method in ['stats'] and field in ['keys','key'] and param in ['markersize']:
						values = [20]
						values = {'fit':values,'predict':values,'interpolate':values}.get(typed,values)
						value = values[i_classes%len(values)]
						params[param] = value
					elif method in ['stats'] and field in ['keys','key'] and param in ['zorder']:
						values = [100]
						values = {'fit':values,'predict':[0],'interpolate':values}.get(typed,values)
						value = values[i_classes%len(values)]
						params[param] = value



	return params

def plotter(data,metadata,settings,models,texify=None,verbose=False):
	if texify is None:
		tex = Texify(**settings['texify'])
		texify = tex.texify


	if not all([key in settings['plot']['settings'] for key in data]):
				settings['plot']['settings'] = {key: {name: copy.deepcopy(settings['plot']['settings'][name]) 
														for name in settings['plot']['names'] if name in settings['plot']['settings']}
												for key in data}	
	names = [name for name in settings['plot']['names'] 
				if name in settings['plot']['fig'] and all([name in settings['plot']['settings'][k] for k in settings['plot']['settings']])
			]




	labels = list(models)
	keys = list(data)
	types = settings['fit']['types']



	fig = settings['plot']['fig']
	axes = settings['plot']['axes']


	obj_update = lambda obj,default,constraints: obj.update({k: copy.deepcopy(default)  for k in obj if all([constraint.get(k) == 0 for constraint in constraints])})
	for ilabel,label in enumerate(labels):
		dims = np.arange(max([len(metadata[key]['rhs_lhs'].get(label,{}).get('lhs',[])) for key in data]))[settings['plot'].get('dims',slice(None))]

		if settings['plot'].get('retain').get('label') is not None:
			obj_update(fig,{},[settings['plot']['retain']['label'],settings['plot']['retain']['fig'],settings['plot']['retain']['axes']])
			obj_update(axes,{},[settings['plot']['retain']['label'],settings['plot']['retain']['fig'],settings['plot']['retain']['axes']])			
		for idim,dim in enumerate(dims):
			if settings['plot'].get('retain').get('dim') is not None:
				obj_update(fig,{},[settings['plot']['retain']['dim'],settings['plot']['retain']['fig'],settings['plot']['retain']['axes']])
				obj_update(axes,{},[settings['plot']['retain']['dim'],settings['plot']['retain']['fig'],settings['plot']['retain']['axes']])
			for ikey,key in enumerate(keys):				
				if settings['plot'].get('retain',{}).get('key') is not None:
					obj_update(fig,{},[settings['plot']['retain']['key'],settings['plot']['retain']['fig'],settings['plot']['retain']['axes']])
					obj_update(axes,{},[settings['plot']['retain']['key'],settings['plot']['retain']['fig'],settings['plot']['retain']['axes']])
				
				if label not in metadata[key]['rhs_lhs']:
					continue

				r = metadata[key]['rhs_lhs'][label]['rhs']
				l = metadata[key]['rhs_lhs'][label]['lhs']

				for iname,name in enumerate(names):

					for ityped,typed in enumerate(types):
					
						first,last = [(((settings['plot'].get('retain',{}).get('fig',{}).get(name,False)) 
										# and
										# (not settings['plot'].get('retain',{}).get('dim',{}).get(name,False)) and 
										# (not settings['plot'].get('retain',{}).get('key',{}).get(name,False)) 
										) and  
										((ilabel==jlabel) and (idim==jdim) and (ikey==jkey) and 
										 (iname==jname) and (ityped==jtyped)))
									  for jlabel,jdim,jkey,jname,jtyped in zip(
											[0,len(labels)-1],[0,len(dims)-1],[0,len(keys)-1],
											[0,len(names)-1],[0,len(types)-1])
									]

						label_dim = DELIMITER.join([label,typed,'%r'%(dim)])

						if ((name in settings['plot']['groups']['fit']) and 
							((typed not in metadata[key]) or 
							 (label_dim not in metadata[key][typed]))):
							continue

						if (((not first) and (typed in ['fit']) and settings['plot'].get('first_last',False))):
							continue


						if ((not ((name in settings['plot']['groups']['fit']) and 
								((settings['plot']['settings'][key][name]['other'].get('y') is None) or 
								(l[dim] in settings['plot']['settings'][key][name]['other']['y'])))) and 
							(((name in settings['plot']['groups']['plot']) and
								  ((ilabel > 0) or (ityped > 0))))):
							continue


						logger.log(verbose,'Plotting %s %s %s'%(l[dim],name,key))

						if name in settings['plot']['groups']['fit']:
							stats = metadata[key][typed][label_dim]['stats']
							iloc = {
								'indiv':metadata[key]['iloc'][dim%len(metadata[key]['iloc'])] if metadata[key]['iloc'] not in [None] else metadata[key]['iloc'],
								'multi':metadata[key]['iloc'][dim%len(metadata[key]['iloc'])] if metadata[key]['iloc'] not in [None] else metadata[key]['iloc'],
								}[settings['model']['approach']] 
						else:
							stats = {}
							iloc = None


						file = settings['sys']['files']['plot']

						fields = {}
						field = 'iloc'
						if ((metadata[key][field] not in [None]) and (len(metadata[key][field]) > 1)):
							fields[field] = str(iloc) if isinstance(iloc,(int,np.integer)) else str(None)
						if len([typed for typed in types if typed in metadata[key]])>1:
							field = 'type'
							fields[field] = typed

						fields = [DELIMITER.join([field,fields[field]]) for field in fields]

						if name in settings['plot']['groups']['fit']:
							file = file%(name,DELIMITER.join([l[dim],*fields])) if (file.count('%s') == 2) else file%(name) if (file.count('%s') == 1) else file
						elif name in settings['plot']['groups']['plot']:
							file = file%(name,name) if (file.count('%s') == 2) else file%(DELIMITER.join([name,'%s'])) if (file.count('%s') == 1) else file
						else:
							file = file%(name,DELIMITER.join([l[dim],*fields])) if (file.count('%s') == 2) else file%(name) if (file.count('%s') == 1) else file
						# else:
						# 	if name in settings['plot']['groups']['fit']:
						# 		file = DELIMITER.join(['plot',name,'%s%s'%(l[dim],settings['sys']['identity'])]) 
						# 	elif name in settings['plot']['groups']['plot']:     
						# 		file = DELIMITER.join(['plot',name,'%s%s'%(l[dim],settings['sys']['identity'])])  
						# 	else:
						# 		file = name                                           
						# 	ext = settings['sys']['ext']['plot']

						path = path_join(metadata[key]['directory']['dump'],file,ext=settings['sys']['ext']['plot'])
						mplstyle = settings['plot']['mplstyle'][name]

						if name == 'Loss':

							thresholds = {}
							# thresholds['derivative_polynomial'] = {i:o for i,o in enumerate(stats['ordering'][::-1]) if any([u.startswith('partial') and int(u.split('_')[-1])>0 for u in o.split('-')])}
							# thresholds['polynomial_polynomial'] = {i:o for i,o in enumerate(stats['ordering'][::-1]) if all([(u.startswith('partial') and int(u.split('_')[-1])==0) or (not u.startswith('partial')) for u in o.split('-')])}
							# thresholds['derivative'] = {i:o for i,o in enumerate(stats['ordering'][::-1]) if o.startswith('partial')}
							for k in list(thresholds):
								if len(thresholds[k]) == 0:
									thresholds.pop(k);

							_options =  {
								'%s_%s'%(x,y):{
									'fig':{
										'set_size_inches':{'w':25,'h':25},
										# 'subplots_adjust':{'hspace':2,'wspace':2}, #,'left':0.1,'bottom':0.1,'right':0.1,'top':0.1},
										# 'subplots_adjust':{'left':1.5,'top':1.5}, #,'left':0.1,'bottom':0.1,'right':0.1,'top':0.1},
										'tight_layout':{},#'pad':1000,'h_pad':500,'w_pad':500},								   			
										'savefig':{'fname':path,'bbox_inches':'tight'},
										**({'close':{'fig':None}} if last else {}),
										},
									'ax':{
										'plot':{
											'x': stats[x],
											'y': np.array(stats[y])*(stats['scale']['y'][dim]**(settings['plot']['rescale'][name] - (not stats['rescale']['loss']))),
											# 'label':r'{{\mathcal{D}}_{\textrm{%s}}}'%('data'),	
											'marker': 's',
											'fillstyle': 'full',
											'linestyle': '-',
											'markersize': 20,
											'linewidth': 5,
											'alpha': 0.7,
											'zorder': 100,
											'markeredgewidth': 6,
											**_get_plot(x,y,key,typed,label_dim,name,metadata,constant=settings['plot']['settings'][key][name]['other']['constant']),
										},
										# 'axvline':[{'x':f(thresholds[k]),'ymin':0.25,'ymax':0.75,
										# 			'linestyle':'--',
										# 			'label':thresholds[k][f(thresholds[k])],
										# 			'color':{'derivative_polynomial':'k',
										# 					 'polynomial_polynomial':'r'}[k]} 
										# 			for k in thresholds for f in [max,min]],
										'set_aspect':{'aspect':'auto'},
										'set_title':{'label':' ','pad':1,'fontsize':settings['plot']['settings'][key][name]['other']['fontsize']},
										'set_ylabel':{'ylabel':'loss','fontsize':settings['plot']['settings'][key][name]['other']['fontsize']},
										'set_xlabel':{'xlabel':'complexity_','fontsize':settings['plot']['settings'][key][name]['other']['fontsize']},
										# 'set_yoffsetText_fontsize':{'fontsize':36},											
										# 'set_ynbins':{'nbins':5},
										'set_xlim':{'xmin':stats['size'],'xmax':0},
										# 'set_ylim':{'ymin':min(1e-4,min(stats['loss'])),
										# 			'ymax':max(1e-1,min(stats['loss']))},
										'set_ylim':{'ymin':1e-7,'ymax':1e0},														

										'set_yscale':{'value':'log','base':10},


										'set_xticks':{'ticks':np.linspace(0,(int(stats['size']/10)+1)*10,4).astype(int).tolist()},
										# 'minorticks_off':{},										
										'tick_params':[
											# {'axis':'x','which':'minor','labelsize':0,'length':15,'width':5},
											{'axis':'y','which':'major','labelsize':settings['plot']['settings'][key][name]['other']['fontsize'],'length':25,'width':5},
											{'axis':'y','which':'minor','labelsize':settings['plot']['settings'][key][name]['other']['fontsize'],'length':15,'width':5},
											{'axis':'x','which':'major','labelsize':settings['plot']['settings'][key][name]['other']['fontsize'],'length':25,'width':5},
											],
										# 'set_ymajor_formatter':{'ticker':'ScalarFormatter'},
							   # 			'ticklabel_format':{'axis':'y','style':'sci','scilimits':[-1,1]},
										'legend':{
											'title_fontsize': 32,
											'prop': {'size': 20},
											'markerscale': 1.2,
											'handlelength': 3,
											'framealpha': 0.8,
											'loc': 'lower right',
											'ncol': 2,
											# 'zorder':100,
											},
										# 'legend':{
										# 	'title_fontsize': 20,
										# 	'prop': {'size': 10},
										# 	'markerscale': 1.2,
										# 	'handlelength': 3,
										# 	'framealpha': 0.8,
										# 	'loc': 'upper left',
										# 	'ncol': 2,
										# 	# 'zorder':100,
										# 	},											
									},
									'style':{
										'texify':texify,
										'mplstyle':settings['plot']['mplstyle'][name],	
										'rcParams':{'font.size':settings['plot']['settings'][key][name]['other']['fontsize']},
										'layout':{'nrows':1,'ncols':1,'index':1},
										'share': {'ax':{'legend':{'title':True,'handles':True,'labels':True}}}
									}
								}
								for x,y in zip(['complexity_'],['loss'])
								}

						elif name == 'BestFit':

							iterations = [(i if i is not None and i>=0 else stats['size_'].max()+1-i if i is not None else stats['size_'].max()) 
											for i in settings['plot']['settings'][key][name]['other']['iterations'] if ((i is None) or ((i <= stats['size_'].max()) and '%s%s%d'%(label_dim,DELIMITER,i) in data[key]))]
							iterations = list(sorted(set(iterations),key=lambda i: iterations.index(i)))
							directory,file,ext = path_split(path,directory=True,file=True,ext=True)
							path = path_join(directory,'_'.join([str(f) for f in [file,*iterations]]),ext=ext)


							_X = [x for x in settings['plot']['settings'][key][name]['other']['x']]
							_Y = [y for y in ([l[dim],*iterations] if settings['plot']['settings'][key][name]['other']['data'] else iterations)
								  if ('%s%s%d'%(label_dim,DELIMITER,y) if isinstance(y,(int,np.integer)) else y) in data[key]]

							X = {x: data[key][x].values*metadata[key]['scale'].get(x,1) if x in data[key] else data[key].index.values
									for x in _X
								}
							Y = {(x,y): data[key]['%s%s%d'%(label_dim,DELIMITER,y) if isinstance(y,(int,np.integer)) else y].values*(stats['scale']['y'][dim]**(settings['plot']['rescale'][name] - (not stats['rescale']['predict'])))
								for y in _Y for x in _X
								}

							if settings['plot']['settings'][key][name]['other']['sort']:
								indices = {x: X[x].argsort() for x in X}
							else:
								indices = {x: slice(None) for x in X}

							X.update({x: X[x][indices[x]] for x in X})
							Y.update({(x,y): Y[(x,y)][indices[x]] for (x,y) in Y})

							_options =  {
								str((y,x)):{
									'fig':{
										'set_size_inches':{'w':15,'h':15},
										# 'subplots_adjust':{'wspace':2,'hspace':2},
										# 'tight_layout':{'pad':300,'h_pad':100},								   			
										'tight_layout':{},								   			
										'savefig':{'fname':path,'bbox_inches':'tight'},
										**({'close':{'fig':None}} if last else {}),
										},
									'ax':{
										'plot':{
											'x':X[x][:-1],
											'y':Y[(x,y)][:-1],
											'label':((DELIMITER.join(['iteration',str(y),DELIMITER.join(['subscript',str(iloc),l[dim].replace('partial','delta')])]) if isinstance(y,(int,np.integer)) else y
													) if ((iloc is not None) and (len(metadata[key]['iloc'])>1)) else (
													DELIMITER.join(['iteration',str(y),l[dim].replace('partial','delta')]) if isinstance(y,(int,np.integer)) else l[dim]
													)),
											'marker':{l[dim]:'',**{_t:_m for _t,_m in zip(iterations,['o','s','*','^','d','v','>','<'])}}.get(y,''),
											'markersize':{l[dim]:3 if Y[(x,y)].shape[0]>1000 else 6,**{_t:3 if data[key].shape[0]>1000 else 6 for _t,_m in zip(iterations,['o','s','*','^','d','v','>','<'])}}.get(y,''),
											# 'markeredgewidth':{l[dim]:4 if Y[(x,y)].shape[0]>1000 else 4,**{_t:1 if data[key].shape[0]>1000 else 1 for _t,_m in zip(iterations,['o','s','*','^','d','v','>','<'])}}.get(y,''),
											# 'color':plt.get_cmap('tab10')(i),
											'color':None,
											'linewidth':(2.5  if data[key].shape[0]>1000 else 4) if y!=l[dim] else 4,
											'linestyle':{l[dim]:'--',**{_t:None for _t in iterations}}.get(y,),
											'fillstyle':'full',
											'alpha':{l[dim]:0.6,**{_t:0.7 for _t in iterations}}.get(y,),
											'zorder':iterations.index(y)+1 if y in iterations else 0,										     
										},
										'set_title':{'label':None,'pad':50,'fontsize':settings['plot']['settings'][key][name]['other']['fontsize']},
										'set_ylabel':{'ylabel':'%s%s'%('',l[dim].replace('partial','partial')),'fontsize':settings['plot']['settings'][key][name]['other']['fontsize']},
										'set_xlabel':{'xlabel':'%s'%(x),'fontsize':settings['plot']['settings'][key][name]['other']['fontsize']},
										'set_yoffsetText_fontsize':{'fontsize':int(settings['plot']['settings'][key][name]['other']['fontsize']*2/3)},
										'set_ynbins':{'nbins':5},
										'set_ymajor_formatter':{'ticker':'ScalarFormatter'},											
										'ticklabel_format':{'axis':'y','style':'sci','scilimits':[-1,2]},
										'tick_params':[
											{'axis':'x','which':'major','labelsize':settings['plot']['settings'][key][name]['other']['fontsize'],'length':10,'width':2},
											{'axis':'y','which':'major','labelsize':settings['plot']['settings'][key][name]['other']['fontsize'],'length':10,'width':2},
											],
										'legend':{'loc':'best','prop':{'size':int(settings['plot']['settings'][key][name]['other']['fontsize']*3/4)},'markerscale':1.5,},
									},
									'style':{
										'texify':texify,
										'mplstyle':settings['plot']['mplstyle'][name],	
										'rcParams':{'font.size':settings['plot']['settings'][key][name]['other']['fontsize']},	        									        		
										'layout':{'nrows':1,'ncols':1,'index':1},
									}
								}
								for i,y in enumerate(_Y)
								for j,x in enumerate(_X) 
								}


						elif name == 'Coef':



							ordering = stats['ordering']
							complexity = stats['complexity_']
							ncols = settings['plot']['settings'][key][name]['style']['layout']['ncols']
							nrows = settings['plot']['settings'][key][name]['style']['layout']['nrows']
							N = ncols*nrows

							Nslice = slice(-min(len(ordering),len(complexity),N),None)
							N = len(ordering[Nslice]) #len(ordering)
							complexity = [*complexity,*[complexity[-1] for _ in range(N-len(complexity))]]
							coef_ = stats['coef_']

							replacements = {}
							replacements.update({'partial':'delta'})
							# replacements.update({'taylorseries%s'%(DELIMITER):'','expansion%s'%(DELIMITER):''})
							replacements.update({'taylorseries':'variable','constant':'variable'})
							# replacements.update({'expansion':'derivative','constant':'derivative'})



							# for i,y in enumerate(zip(ordering[-1:-10-1:-1],complexity[-1:-10-1:-1])):
							# 	print(y[0])
							# 	print(texify(_replace(y[0],replacements)))
							# 	print()

							_options =  {
								str(y):{
									'fig':{
										'set_size_inches':{'w':30,'h':30},
										'subplots_adjust':{'wspace':0.5,'hspace':0.5},
										'tight_layout':{'pad':300,'h_pad':100},								   			
										# 'savefig':{'fname':path,'bbox_inches':'tight'},
										'savefig':{'fname':path,'bbox_inches':'tight'},
										**({'close':{'fig':None}} if last else {})
										},
									'ax':{
										'plot':{
											'x':stats['complexity_'],
											'y':np.abs(coef_[y[0]]),
											# 'label':'%s - %s'%(y[0].split(DELIMITER)[2],key.split('/')[1]),
											'marker':'o',
											'fillstyle':'full',
											'markersize':10,
											'linewidth':3,
											'alpha':0.8,
											'zorder':100,								     
										},
										'set_title':{'label':r'%s - %s'%(_replace(texify(_replace(y[0],replacements)),replacements),y[1]),'pad':75,'fontsize':settings['plot']['settings'][key][name]['other']['fontsize'] - (16 if len(texify(y))>10 else 0)},
										'set_xlabel':{'xlabel':'complexity_','fontsize':settings['plot']['settings'][key][name]['other']['fontsize']},
										'set_ylabel':{'ylabel':'coef_','fontsize':settings['plot']['settings'][key][name]['other']['fontsize']},
										'set_yoffsetText_fontsize':{'fontsize':int(settings['plot']['settings'][key][name]['other']['fontsize']*2/3)},
										'set_ynbins':{'nbins':5},
										'set_xlim':{'xmin':stats['size'],'xmax':0},
										'set_xticks':{'ticks':np.linspace(0,(int(stats['size']/10)+1)*10,4).astype(int).tolist()},
										'set_yscale':{'value':'linear'},
										'set_ymajor_formatter':{'ticker':'ScalarFormatter'},											
										'ticklabel_format':{'axis':'y','style':'sci','scilimits':[-2,2]},
										'tick_params':[
											{'axis':'x','which':'major','labelsize':settings['plot']['settings'][key][name]['other']['fontsize'],'length':10,'width':2},
											{'axis':'y','which':'major','labelsize':settings['plot']['settings'][key][name]['other']['fontsize'],'length':10,'width':2},
											],
									},
									'style':{
										'texify':texify,
										'mplstyle':settings['plot']['mplstyle'][name],	
										'rcParams':{'font.size':settings['plot']['settings'][key][name]['other']['fontsize']},						        		
										'layout':{
											'nrows':nrows,
											'ncols':ncols,
											'index':i+1,
										},
										'share':{
											'ax':{
												'set_xlabel':{'xlabel':None},
												'set_ylabel':{'ylabel':'left'},
											}
										}
									}
								}
								for i,y in enumerate(zip(ordering[Nslice],complexity[Nslice]))}

						elif name == 'Error':    
							iterations = [(i if i is not None and i>=0 else stats['size_'].max()+1-i if i is not None else stats['size_'].max()) 
											for i in settings['plot']['settings'][key][name]['other']['iterations'] if ((i is None) or ((i <= stats['size_'].max()) and '%s%s%d'%(label_dim,DELIMITER,i) in data[key]))]
							error = lambda y: np.mean(((data[key][l[dim]]-data[key]['%s%s%d'%(label_dim,DELIMITER,y) if isinstance(y,(int,np.integer)) else y]).values))*(stats['scale']['y'][dim]**(settings['plot']['rescale'][name] - (not stats['rescale']['predict'])))
							directory,file,ext = path_split(path,directory=True,file=True,ext=True)
							path = path_join(directory,'_'.join([str(f) for f in [file,*iterations]]),ext=ext)
							

							_X = [x for x in settings['plot']['settings'][key][name]['other']['x']]
							_Y = [y for y in ([l[dim],*iterations] if settings['plot']['settings'][key][name]['other']['data'] else iterations)
								  if ('%s%s%d'%(label_dim,DELIMITER,y) if isinstance(y,(int,np.integer)) else y) in data[key]]


							_options =  {
								str((y,x)):{
									'fig':{
										'set_size_inches':{'w':25,'h':25},
										'subplots_adjust':{'wspace':2,'hspace':2},
										# 'tight_layout':{'pad':300,'h_pad':100},								   			
										'savefig':{'fname':path,'bbox_inches':'tight'},
										**({'close':{'fig':None}} if last else {})
										},
									'ax':{
										'plot':{
											'x':data[key][x].values*metadata[key]['scale'].get(x,1) if x in data[key]  else data[key].index.values,
											'y':np.array(((data[key][l[dim]]-data[key]['%s%s%d'%(label_dim,DELIMITER,y) if isinstance(y,(int,np.integer)) else y]).values)*(stats['scale']['y'][dim]**(settings['plot']['rescale'][name] - (not stats['rescale']['predict']))))/(data[key][l[dim]].values),
											'label':((DELIMITER.join(['iteration',str(y),DELIMITER.join(['subscript',str(iloc),l[dim].replace('partial','delta')])]) if isinstance(y,(int,np.integer)) else y
													) if ((iloc is not None) and (len(metadata[key]['iloc'])>1)) else (
													DELIMITER.join(['iteration',str(y),l[dim].replace('partial','delta')]) if isinstance(y,(int,np.integer)) else l[dim]
													)),											
											'marker':{l[dim]:'',**{_t:_m for _t,_m in zip(iterations,['o','s','*','^','d',''])}}.get(y,''),
											'markersize':3 if data[key].shape[0]>1000 else 6,
											# 'color':np.random.rand(3),
											'linewidth':1  if data[key].shape[0]>1000 else 3,
											'linestyle':{l[dim]:'--',**{_t:None for _t in iterations}}.get(y,),
											'fillstyle':'full',
											'alpha':{l[dim]:0.6,**{_t:0.8 for _t in iterations}}.get(y,),
											'zorder':iterations.index(y)+1 if y in iterations else 0,										     
										},
										'set_title':{'label':None,'pad':50,'fontsize':settings['plot']['settings'][key][name]['other']['fontsize']},
										'set_ylabel':{'ylabel':r'\textrm{Model Error}','fontsize':settings['plot']['settings'][key][name]['other']['fontsize']},
										'set_xlabel':{'xlabel':'%s'%(x),'fontsize':settings['plot']['settings'][key][name]['other']['fontsize']},
										'set_yoffsetText_fontsize':{'fontsize':int(settings['plot']['settings'][key][name]['other']['fontsize']*2/3)},
										'set_ynbins':{'nbins':5},
										'set_ymajor_formatter':{'ticker':'ScalarFormatter'},											
										'ticklabel_format':{'axis':'y','style':'sci','scilimits':[-1,1]},
										'tick_params':[
											{'axis':'x','which':'major','labelsize':settings['plot']['settings'][key][name]['other']['fontsize'],'length':10,'width':2},
											{'axis':'y','which':'major','labelsize':settings['plot']['settings'][key][name]['other']['fontsize'],'length':10,'width':2},
											],
										'set_yscale':{'value':'log','base':10},
										'legend':{'loc':'best','prop':{'size':int(settings['plot']['settings'][key][name]['other']['fontsize']*3/4)},'markerscale':2,},
									},
									'style':{
										'texify':texify,
										'mplstyle':settings['plot']['mplstyle'][name],	
										'rcParams':{'font.size':settings['plot']['settings'][key][name]['other']['fontsize']},	        									        		
										'layout':{'nrows':1,'ncols':1,'index':1},
									}
								}
								for y in _Y
								for x in _X
								}



						elif name == 'Variables': 

							X = [x if isinstance(x,list) else [x] for x in settings['plot']['settings'][key][name]['other']['x']]
							Y = [x if isinstance(x,list) else [x] for x in settings['plot']['settings'][key][name]['other']['y']]

							Xdata = [[data[key][_x].values*metadata[key]['scale'].get(_x,1) if _x in data[key]  else data[key].index.values 
									for _x in x] for x in X]
							Ydata = [[data[key][_y].values*metadata[key]['scale'].get(_y,1) if _y in data[key]  else data[key].index.values 
											for _y in y] for y in Y]									
							Xmin = [np.min(x) for x in Xdata]
							Xmax = [np.max(x) for x in Xdata]
							Ymin = [np.min(y) for y in Ydata]
							Ymax = [np.max(y) for y in Ydata]
							
							Ndata = min(len(X),len(Y))

							subplots = settings['plot']['settings'][key][name]['other']['subplots']

							ncols = max(1,min(Ndata,settings['plot']['settings'][key][name]['style']['layout']['ncols'] if subplots else 1))
							nrows = max(1,min(max(1,int(Ndata/ncols)),settings['plot']['settings'][key][name]['style']['layout']['nrows'] if subplots else 1))
							Nplots = ncols*nrows

							_options =  {
								'%s-%s'%(','.join([_x for _x in x]),
										 ','.join([_y for _y in y])):{
									'fig':{
										'set_size_inches':{'w':13,'h':13},
										# 'subplots_adjust':{'wspace':0.5,'hspace':0.6},
										'subplots_adjust':{'wspace':0.75,'hspace':0.6},
										'savefig':{'fname':path},
										**({'close':{'fig':None}} if last else {}),
										},
									'ax':{
										'plot':[{
											'x':Xdata[t][_t],
											'y':Ydata[t][_t],
											'marker':'^',
											'markersize':6,
											'linewidth':3,
											'linestyle':'-',
											'fillstyle':'full',
											'alpha':0.6,
											'zorder':0,	
											'label':_y,									     
										} for _t,(_x,_y) in enumerate(zip(x,y))],
										'set_title':{'label':'%s%s'%(DELIMITER,os.path.commonprefix(list(set(y)))) if not os.path.commonprefix(list(set(y))).endswith('_') else os.path.commonprefix(list(set(y)))[:-1] if len(set(y))>1 else y[0],'pad':30,'fontsize':settings['plot']['settings'][key][name]['other']['fontsize']},
										'set_ylabel':{'ylabel':'%s%s'%(DELIMITER,' '.join([_y for _y in set(y)])) if len(set(y))==1 else None,'fontsize':settings['plot']['settings'][key][name]['other']['fontsize']},
										'set_xlabel':{'xlabel':'%s'%(' '.join([_x for _x in set(x)])) if len(set(x))==1 else None,'fontsize':settings['plot']['settings'][key][name]['other']['fontsize']},
										'set_yoffsetText_fontsize':{'fontsize':int(settings['plot']['settings'][key][name]['other']['fontsize']*2/3)},
										'set_ynbins':{'nbins':4},
										'set_xnbins':{'nbins':3},
										'set_yticks':{'ticks':np.linspace(Ymin[t]-2,Ymax[t]+3,5)} if all([np.all(np.array(_y,dtype=int)==np.array(_y)) for _y in Ydata[t]]) else None,											
										'set_ymajor_formatter':{'ticker':'ScalarFormatter'},											
										'ticklabel_format':{'axis':'y','style':'sci','scilimits':[-2,2]},
										'minorticks_off':{},
										'tick_params':[
											{'axis':'x','which':'major','labelsize':settings['plot']['settings'][key][name]['other']['fontsize'],'length':10,'width':2},
											{'axis':'y','which':'major','labelsize':settings['plot']['settings'][key][name]['other']['fontsize'],'length':10,'width':2},
											],
										'legend':{'loc':'best','prop':{'size':20},'markerscale':1,} if len(set(y))>1 else None,
									},
									'style':{
										'texify':texify,
										'mplstyle':settings['plot']['mplstyle'][name],
										'rcParams':{'font.size':settings['plot']['settings'][key][name]['other']['fontsize']},											
										'layout':{'nrows':nrows,'ncols':ncols,'index':t+1 if subplots else 1},
										'share':{
											'ax':{
												'set_xlabel':{'xlabel':'bottom'},
												'set_ylabel':{'ylabel':False},
											}
										}
									}
								}
								for t,(x,y) in enumerate(zip(X,Y))
								}



						elif name == 'Operators': 

							continue


							operatore = operators()
							weightore = weights()	
							labelore = labels()		


						
						
							weights = settings['plot']['settings'][key][name]['other']['weights']
							neighbourhood = settings['plot']['settings'][key][name]['other']['neighbour']
							plot_terms = settings['plot']['settings'][key][name]['other']['terms']
							plot_terms = [{'function':['Psi'],'variable':['vol_square'],'order':[1]},
										  {'function':['Psi'],'variable':['vol_square','E11'],'order':[2]}]

							for weight in weights:
								for neighbours in neighbourhood:
									for term in terms:

										if sum((all([(t in c[k] and c[k].index(t)==i)
														for k in c
														for i,t in enumerate((term[k] 
															if isinstance(term[k],list) else [term[k]])) 
														]) for c in constraints))!=1:

											continue

										term = copy.deepcopy(term)

										for key.value in zip(['label','weighting'],[weight,neigbours]):
											term[key] = [value for x in term[key]]
										for key in ['label','weighting']:
											term[key] = labelore[key](**term)


										functions = [term['function'],*term['label']]

										for j in range(term['order']):
											label = functions[j+1]
											weighting = term['weighting'][j]
											if (label in df):
												continue
											function = check(df,functions[j])
											variable = check(df,term['variable'][j])
											variables = check(df,term['variables'][j])
											weight = series_array(df[weighting]) if (weighting in df) else weightore[term['weight'][j]]
											neighbours = check(df,term['adjacency'][j])
											operation = operatore[term['operation'][j]]
											params = copy.deepcopy(parameters)

											_op,_weight = operation(variable,function,variables,
																		 weight,neighbours,params)
											# if label not in df:
											# 	df[label] = _op
											# if weighting not in df:
											# 	df[weighting] = array_series(_weight,index=df.index)

							X = [x if isinstance(x,list) else [x] for x in settings['plot']['settings'][key][name]['other']['x']]
							Y = [x if isinstance(x,list) else [x] for x in settings['plot']['settings'][key][name]['other']['y']]
							coef_ = stats['coef_']
							coef_names = [c.replace(settings['terms']['weight'],'WEIGHT') for c in coef_]

							continue

							Ndata = min(len(X),len(Y))

							subplots = settings['plot']['settings'][key][name]['other']['subplots']

							ncols = max(1,min(Ndata,settings['plot']['settings'][key][name]['style']['layout']['ncols'] if subplots else 1))
							nrows = max(1,min(max(1,int(Ndata/ncols)),settings['plot']['settings'][key][name]['style']['layout']['nrows'] if subplots else 1))
							Nplots = ncols*nrows

							_options =  {
								'%s-%s'%(','.join([_x for _x in x]),
										 ','.join([_y for _y in y])):{
									'fig':{
										'set_size_inches':{'w':13,'h':13},
										# 'subplots_adjust':{'wspace':0.5,'hspace':0.6},
										'subplots_adjust':{'wspace':2,'hspace':0.6},
										'savefig':{'fname':path%(' '.join([_x for _x in x])) if not subplots else path%(name)},
										**({'close':{'fig':None}} if last else {}),
										},
									'ax':{
										'plot':[{
											'x':data[key][_x].values*metadata[key]['scale'].get(_x,1) if _x in data[key]  else data[key].index.values,
											'y':data[key][_y].values*metadata[key]['scale'].get(_y,1)*(stats['scale']['y'][dim]**(settings['plot']['rescale'][name] - (not stats['rescale']['coef_']))) if _y in data[key]  else data[key].index.values,
											'marker':'^',
											'markersize':6,
											'linewidth':3,
											'linestyle':'-',
											'fillstyle':'full',
											'alpha':0.6,
											'zorder':0,	
											'label':_y,									     
										} for _x,_y in zip(x,y)],
										'set_title':{'label':'%s%s'%(DELIMITER,os.path.commonprefix(list(set(y)))) if len(set(y))>1 else y[0],'pad':20,'fontsize':settings['plot']['settings'][key][name]['other']['fontsize']},
										'set_ylabel':{'ylabel':'%s%s'%(DELIMITER,' '.join([_y for _y in set(y)])) if len(set(y))==1 else None,'fontsize':settings['plot']['settings'][key][name]['other']['fontsize']},
										'set_xlabel':{'xlabel':'%s'%(' '.join([_x for _x in set(x)])) if len(set(x))==1 else None,'fontsize':settings['plot']['settings'][key][name]['other']['fontsize']},
										'set_yoffsetText_fontsize':{'fontsize':int(settings['plot']['settings'][key][name]['other']['fontsize']*2/3)},
										'set_ynbins':{'nbins':4},
										'set_xnbins':{'nbins':3},
										'set_ymajor_formatter':{'ticker':'ScalarFormatter'},											
										'ticklabel_format':{'axis':'y','style':'sci','scilimits':[-1,1]},
										'minorticks_off':{},											
										'tick_params':[
											{'axis':'x','which':'major','labelsize':settings['plot']['settings'][key][name]['other']['fontsize'],'length':10,'width':2},
											{'axis':'y','which':'major','labelsize':settings['plot']['settings'][key][name]['other']['fontsize'],'length':10,'width':2},
											],
										'legend':{'loc':'best','prop':{'size':20},'markerscale':1,} if len(set(y))>1 else None,
									},
									'style':{
										'texify':texify,
										'mplstyle':settings['plot']['mplstyle'][name],
										'rcParams':{'font.size':settings['plot']['settings'][key][name]['other']['fontsize']},											
										'layout':{'nrows':nrows,'ncols':ncols,'index':t+1 if subplots else 1},
										'share':{
											'ax':{
												'set_xlabel':{'xlabel':'bottom'},
												'set_ylabel':{'ylabel':False},
											}
										}
									}
								}
								for t,(x,y) in enumerate(zip(X,Y))
								}




						# Update plot settings, in order of preference of {settings['plot']['settings'][key][name] (from settings),
						# 												   options (from file),
						# 												   _options (from main)}
						# Typicallly, settings['plot']['settings'][key][name]['other'] and options (from file) 
						# contain standard props for each prop key to be shared between all prop keys


						_options_ = [load(settings['plot']['file'][name],default={}),
									 copy.deepcopy(settings['plot']['settings'][key][name])]
						options = copy.deepcopy(_options)


						for opts in _options_:
							if not all([k in opts for k in options]):
								opts = {k:copy.deepcopy(opts) for k in options}
							_update(options,opts,_clear=False,
										_func=lambda k,i,e: (_formatstring(k,i,e,
											**{
												**{attr:[key if ((ikey>0 or (len(keys)==1)) or typed not in ['fit']) else {'label':
																											# r'%s'%(r'data')
																											r'%s'%(r'{\mathcal{D}}_{0-%d}'%(len(keys)-1))
																											}.get(attr,''),
														'%r'%((list(set(iloc)) [0] if ((not isinstance(iloc,(int,np.integer))) and (iloc is not None)) else iloc) if (
														(isinstance(iloc,(int,np.integer))) or (len(iloc)>1)) else None)]
													for attr in ['label']},
												# **{attr: [r'%s'%(matplotlib.colors.to_hex(plt.get_cmap('tab10')(np.linspace(0, 1,max(1,len(metadata[key]['iloc']))))[
												# 	metadata[key]['iloc'].index(iloc)]))]
												# 	for attr in ['color']},
											}) if ( (k in [
													'label',
													# 'color'
													]) and (e is not None and k in e) and ((i is not None and k not in i) or (not all([i[k] is not None and t in i[k] for t in ['%s%s'%(DELIMITER,l[dim])]])) or 
												   ( e[k] in ['%s']) or  (e[k] is not None and ('%s') in e[k]))) else (e[k]))
										)

						
						fig[name],axes[name] = plot(settings=options,fig=fig[name],axes=axes[name],
													texify=texify,mplstyle=mplstyle)



