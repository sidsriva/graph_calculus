#!/usr/bin/env python

# Import python modules
import os,sys,copy,warnings,itertools,inspect
import json,glob
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker

# Global Variables
CLUSTER = 0
if CLUSTER:
	PATH = '~/graph/mechanochem-ml-code'
else:
	PATH = '~/files/um/code/mechanochem-ml-code'

DELIMITER='__'
MAX_PROCESSES = 1
PARALLEL = 0

# Import user modules
from utility.texify import Texify

warnings.simplefilter('ignore', (UserWarning,DeprecationWarning,FutureWarning))

# Update nested elements
def _update(iterable,elements,_copy=False,_clear=True,_func=None):
	if not callable(_func):
		_func = lambda key,iterable,elements: elements[key]
	if _clear and elements == {}:
		iterable.clear()
	if not isinstance(elements,(dict)):
		iterable = elements
		return
	for e in elements:
		if isinstance(iterable.get(e),dict):
			if e not in iterable:
				iterable.update({e: elements[e]})
			else:
				_update(iterable[e],elements[e],_copy=_copy,_clear=_clear,_func=_func)
		else:
			iterable.update({e:elements[e]})
	return


# List from generator
def list_from_generator(generator,field=None):
	item = next(generator)
	if field is not None:
		item = item[field]    
	items = [item]
	for item in generator:
		if field is not None:
			item = item[field]
		if item == items[0]:
			break
		items.append(item)

	# Reset iterator state:
	for item in generator:
		if field is not None:
			item = item[field]
		if item == items[-1]:
			break
	return items

# Check if obj is number
def is_number(s):
	try:
		s = float(s)
		return True
	except:
		try:
			s = int(s)
			return True
		except:
			return False

# Plot data - General plotter
def plot(x=None,y=None,settings={},fig=None,axes=None,mplstyle=None,texify=None,quiet=True):

	AXIS = ['x','y','z']
	AXES = ['colorbar']
	LAYOUT = ['nrows','ncols','index']
	DIM = 2

	def _layout(settings):
		if isinstance(settings,(list,tuple)):
			return dict(zip(LAYOUT,settings))
		_layout_ = {}
		if all([k in settings for k in ['pos']]):
			pos = settings.pop('pos')
			if pos not in [None]:
				pos = str(pos)
				_layout_ = {k: int(pos[i]) for i,k in zip(range(len(pos)),LAYOUT)}
		elif all([k in settings and settings.get(k) not in [None] for k in LAYOUT]):
			_layout_ = {k: settings[k] for k in LAYOUT}
		if _layout_ != {}:
			settings.update(_layout_)
		else:
			settings.clear()
		return _layout_

	def _position(layout):
		if all([s == t for s,t in zip(LAYOUT,['nrows','ncols'])]):
			position = ((((layout['index']-1)//layout['ncols'])%layout['nrows'])+1,((layout['index']-1)%layout['ncols'])+1)
		else:
			position = (1,1)
		return position

	def _positions(layout):
		if all([s == t for s,t in zip(LAYOUT,['nrows','ncols'])]):
			positions = {
				'top':(1,None),'bottom':(layout['nrows'],None),
				'left':(None,1),'right':(None,layout['ncols']),
				'top_left':(1,1),'bottom_right':(layout['nrows'],layout['ncols']),
				'top_right':(1,layout['ncols']),'bottom_left':(layout['nrows'],1),
				}
		else:
			positions = {
				'top':(1,None),'bottom':(1,None),
				'left':(None,1),'right':(None,1),
				'top_left':(1,1),'bottom_right':(1,1),
				'top_right':(1,1),'bottom_left':(1,1),
				}
		return positions


	def layout(key,fig,axes,settings):
		if all([key in obj for obj in [fig,axes]]):
			return
		_layout_ = _layout(settings[key]['style']['layout'])
		add_subplot = True and (_layout_ != {})
		other = {'%s_%s'%(key,k):settings[key]['style'].get(k) for k in AXES if isinstance(settings[key]['style'].get(k),dict)}
		for k in axes:
			__layout__ = _layout(settings.get(k,{}).get('style',{}).get('layout',axes[k].get_geometry()))
			if all([_layout_[s]==__layout__[s] for s in _layout_]):
				axes[key] = axes[k]
				add_subplot = False
				break

		if fig.get(key) is None:
			if (fig == {} or settings[key]['style'].get('unique_fig',False)) and not hasattr(axes.get(key),'figure'):
				fig[key] = plt.figure()
			elif hasattr(axes.get(key),'figure'):
				fig[key] = getattr(axes.get(key),'figure')
			else:
				k = list(fig)[0]
				fig[key] = fig[k]

		if add_subplot:					
			args = [_layout_.pop(s) for s in LAYOUT]
			gs = gridspec.GridSpec(*args[:DIM])
			axes[key] = fig[key].add_subplot(list(gs)[args[-1]-1],**_layout_)


			for k in other:
				axes[k] = fig[key].add_axes(**other[k])
		return

	def attr_texify(string,attr,kwarg,texify,**kwargs):
		def texwrapper(string):
			s = string.replace('$','')
			if not any([t in s for t in [r'\textrm','_','^','\\']]):
				pass
				# s = r'\textrm{%s}'%s
			# for t in ['_','^']:
			# 	s = s.split(t)
			# 	s = [r'\textrm{%s}'%i  if (not (is_number(i) or any([j in i for j in ['$','textrm','_','^','\\','}','{']]))) else i for i in s]
			# 	s = t.join(['{%s}'%i for i in s])
			s = r'$%s$'%(s)
			return s
		attrs = {
			**{'set_%slabel'%(axis):['%slabel'%(axis)]
				for axis in AXIS},
			# **{'set_%sticks'%(axis):['ticks']
			# 	for axis in AXIS},				
			**{'set_%sticklabels'%(axis):['labels']
				for axis in AXIS},	
			**{k:['label'] for k in ['plot','scatter','axvline','axhline','vlines','hlines','plot_surface']},								
			**{'set_title':['label'],'suptitle':['t'],
			'annotate':['s'],
			'legend':['title']},
		}
		if texify is None:
			texify = texwrapper
		elif isinstance(texify,dict):
			Tex = Texify(**texify)
			texify = Tex.texify
			texify = lambda string,texify=texify: texwrapper(texify(string))


		if attr in attrs and kwarg in attrs[attr]:
			if isinstance(string,(str,tuple)):
				string = texify(string)
			elif isinstance(string,list):
				string = [texify(s) for s in string]
		return string


	def attr_share(value,attr,kwarg,share,**kwargs):
		
		attrs = {
			**{'set_%s'%(key):['%s'%(label)]
				for axis in AXIS 
				for key,label in [('%slabel'%(axis),'%slabel'%(axis)),
								  ('%sticks'%(axis),'ticks'),
								  ('%sticklabels'%(axis),'labels')]},
			**{k:['label'] for k in ['plot','scatter','axvline','axhline','vlines','hlines','plot_surface']},	
			**{
				'set_title':['label'],
				'suptitle':['t'],
				'annotate':['s'],
				'legend':['handles','labels','title']},
		}					
		if ((attr in attrs) and (attr in share) and (kwarg in attrs[attr]) and (kwarg in share[attr])):
			share = share[attr][kwarg]
			if ((share is None) or 
				(not all([(k in kwargs and kwargs[k] is not None) 
					for k in ['layout']]))):
				return value
			elif isinstance(share,bool) and (not share) and (share is not None):
				if isinstance(value,list):
					return []
				else:
					return None     
			elif isinstance(share,bool) and share:
				_position_ = _position(kwargs['layout']) 
				position = _position(kwargs['layout'])
				if all([((_position_[i] is None) or (position[i]==_position_[i])) for i in range(DIM)]):
					return value
				else:
					if isinstance(value,list):
						return []
					else:
						return None     
			else:
				_position_ = _positions(kwargs['layout'])[share]
				position = _position(kwargs['layout'])
				if all([((_position_[i] is None) or (position[i]==_position_[i])) for i in range(DIM)]):
					return value
				else:
					if isinstance(value,list):
						return []
					else:
						return None     						

		else:
			return value
		return

	def attr_wrap(obj,attr,settings,**kwargs):

		def attrs(obj,attr,_attr,**kwargs):
			call = True
			args = []
			kwds = {}
			_args = []
			_kwds = {}
			if attr in ['legend']:
				handles,labels = getattr(obj,'get_legend_handles_labels')()
				# kwargs.update({k: attr_share(attr_texify(v,attr,k,**kwargs),attr,k,**kwargs)  
				# 		for k,v in zip(['handles','labels'],
				# 						getattr(obj,'get_legend_handles_labels')())
				# 		})
				if handles == [] or all([kwargs[k] is None for k in kwargs]):
					call = False
				else:
					kwargs.update(dict(zip(['handles','labels'],[handles,labels])))
				_kwds.update({
					'set_zorder':{'level':100},
					'set_title':{**({'title': kwargs.pop('title',None),'prop':{'size':kwargs.get('prop',{}).get('size')}} 
									if 'title' in kwargs else {'title':None})},
					})
			
			elif attr in ['plot','axvline','axhline']:
				fields = ['color']
				for field in fields:
					# try:
						if kwargs.get(field) == '__cycle__':
							try:
								_obj = _attr[-1]
							except:
								_obj = _attr
							values = list_from_generator(getattr(getattr(obj,'_get_lines'),'prop_cycler'),field)
							kwargs[field] = values[-1]
						
						elif kwargs.get(field) == '__lines__':
							_obj = getattr(obj,'get_lines')()[-1]
							kwargs[field] = getattr(_obj,'get_%s'%(field))()
						
						else:
							continue
					# except:
					# 	kwargs.pop(field)
					# 	pass

				args.extend([kwargs.pop(k) for k in ['x','y'] if kwargs.get(k) is not None])

			elif attr in ['plot_surface','contour','contourf','scatter']:
				args.extend([kwargs.pop(k) for k in ['x','y','z'] if kwargs.get(k) is not None])

			elif attr in ['set_%smajor_formatter'%(axis) for axis in AXIS]:
				axis = attr.replace('set_','').replace('major_formatter','')
				for k in kwargs:
					getattr(getattr(obj,'get_%saxis'%(axis))(),'set_major_formatter')(
						getattr(getattr(matplotlib,k),kwargs[k])())
				call = False

			elif attr in ['tick_params']:
				for axis in AXIS:
					if kwargs['axis'] == axis:
						if kwargs['which'] in ['minor']:
							# formatter = ticker.LogFormatter(minor_thresholds=(10, 0.4))
							# getattr(getattr(obj,'get_%saxis'%(axis))(),'set_%s_formatter'%(kwargs['which']))(formatter)
							continue
							locator = ticker.LogLocator(base=10.0, subs=(0.1,0.2,0.4,0.6,0.8,1,2,4,6,8,10 )) 
							formatter = ticker.LogFormatter(labelOnlyBase=False, minor_thresholds=(1, 0.4))
							locator = ticker.AutoMinorLocator(10)
							# formatter = ticker.NullFormatter()
							getattr(getattr(obj,'get_%saxis'%(axis))(),'set_%s_locator'%(kwargs['which']))(locator)
							getattr(getattr(obj,'get_%saxis'%(axis))(),'set_%s_formatter'%(kwargs['which']))(formatter)
							call = False

			elif attr in ['set_%snbins'%(axis) for axis in AXIS]:
				axis = attr.replace('set_','').replace('nbins','')
				try:
					locator = 'MaxNLocator'
					locator = getattr(plt,locator)(**kwargs)
				except:
					locator = 'LogLocator'
					locator = getattr(plt,locator)(**kwargs)

				getattr(getattr(obj,'%saxis'%(axis)),'set_major_locator')(locator)
				call = False

			elif attr in ['set_%soffsetText_fontsize'%(axis) for axis in AXIS]:
				axis = attr.replace('set_','').replace('offsetText_fontsize','')
				getattr(getattr(getattr(obj,'%saxis'%(axis)),'offsetText'),'set_fontsize')(**kwargs)
				call = False


			elif attr in ['set_colorbar']:
				values = kwargs.get('values')
				colors = kwargs.get('colors')
				norm = matplotlib.colors.Normalize(vmin=min(values), vmax=max(values))  
				normed_values = norm(values)
				cmap = matplotlib.colors.LinearSegmentedColormap.from_list('colorbar', list(zip(normed_values,colors)), N=len(normed_vals)*10)  
				colorbar = matplotlib.colorbar.ColorbarBase(cax=obj, cmap=cmap, norm=norm, orientation='vertical')
				obj = colorbar
				call = True


			elif attr in ['close']:
				try:
					plt.close(**kwargs)
				except:
					plt.close()
				call = False
				
			if not call:	
				return

			_obj = obj
			for a in attr.split('.'):
				try:
					_obj = getattr(_obj,a)
				except:
					break			
			if args != []:
				_attr = _obj(*args,**kwargs)

			else:
				_attr = _obj(**kwargs)

			for k in _kwds:
				getattr(_attr,k)(**_kwds[k])
				

			# except:
			# 	_kwargs = inspect.getfullargspec(getattr(obj,attr))[0]
			# 	args.extend([kwargs[k] for k in kwargs if k not in _kwargs])
			# 	kwargs = {k:kwargs[k] for k in kwargs if k in _kwargs}
			# 	try:
			# 		getattr(obj,attr)(*args,**kwargs)
			# 	except:
			# 		pass
			return _attr

		_kwargs = []
		_wrapper = lambda kwarg,attr,**kwargs:{k: attr_share(attr_texify(kwarg[k],attr,k,**kwargs),attr,k,**kwargs) for k in kwarg}
		_attr = None
		if isinstance(settings,list):
			_kwargs.extend(settings)
		elif isinstance(settings,dict):
			_kwargs.append(settings)
		else:
			return
		for _kwarg in _kwargs:
			_kwds = _wrapper(_kwarg,attr,**kwargs)
			_attr = attrs(obj,attr,_attr,**_kwds)
		return

	def obj_wrap(attr,key,fig,axes,settings):
		attr_kwargs = lambda attr,key,settings:{
			'texify':settings[key]['style'].get('texify'),
			'share':settings[key]['style'].get('share',{}).get(attr,{}),
			'layout':_layout(settings[key]['style'].get('layout',{})),
			}	
		
		matplotlib.rcParams.update(settings[key]['style'].get('rcParams',{}))


		objs = lambda attr,key,fig,axes: {'fig':fig.get(key),'ax':axes.get(key),**{'%s_%s'%('ax',k):axes.get('%s_%s'%(key,k)) for k in AXES}}[attr]
		obj = objs(attr,key,fig,axes)

		exceptions = {
			**{
				prop: {
					'settings':{'set_%sscale'%(AXES[-1]):{'value':'log'}},
					'kwargs':{kwarg: (lambda settings,prop=prop,kwarg=kwarg,obj=obj: (np.log10(settings[prop][kwarg]))) 
														for kwarg in ['z']},
					'pop':False,
					}
				for prop in ['plot_surface']
				},
			**{
				prop: {
					'settings':{'set_%sscale'%(AXES[-1]):{'value':'log'}},
					'kwargs':{kwarg: (lambda settings,prop=prop,kwarg=kwarg,obj=obj: ([r'$10^{%d}$'%(round(t,-1)) 
														for t in (settings['set_%sticks'%(AXES[-1])]['ticks'] if (
														'set_%sticks'%(AXES[-1]) in settings) else (
														getattr(obj,('set_%sticks'%(AXES[-1])).replace('set','get'))() if (
														hasattr(obj,'set_%sticks'%(AXES[-1]).replace('set','get'))) else [0]))])) 
														for kwarg in ['labels']},
					'pop':False,
					}
				for prop in ['set_%sticklabels'%(AXES[-1])]
				},				
			**{
				prop: {
					'settings':{'set_%sscale'%(AXES[-1]):{'value':'log'}},
					'kwargs':{},
					'pop':True,
					}
				for prop in ['set_%sscale'%(AXES[-1])]
				},	

			}

		ordering = {'close':-1,'savefig':-2}

		if obj is not None:
			props = list(settings[key][attr])
			for prop in ordering:
				if prop in settings[key][attr]:
					if ordering[prop] == -1:
						ordering[prop] = len(props)
					elif ordering[prop] < -1:
						ordering[prop] += 1
					props.insert(ordering[prop],props.pop(props.index(prop)))

			for prop in props:
				kwargs = attr_kwargs(attr,key,settings)
				if prop in exceptions and all([((settings[key][attr][k][l] if (k in settings[key][attr]) else (
														getattr(obj,k.replace('set','get'))() if (
														hasattr(obj,k.replace('set','get'))) else None))==exceptions[prop]['settings'][k][l]) 
														for k in exceptions[prop]['settings'] 
														for l in exceptions[prop]['settings'][k]]):
					for kwarg in exceptions[prop]['kwargs']:
						settings[key][attr][kwarg] = exceptions[prop]['kwargs'][kwarg](settings[key][attr])
					if exceptions[prop]['pop']:
						continue

				attr_wrap(obj,prop,settings[key][attr][prop],**kwargs)
		return
		
		
	def context(x,y,settings,fig,axes,mplstyle,texify):
		with matplotlib.style.context(mplstyle):
			settings,fig,axes = setup(x,y,settings,fig,axes,mplstyle,texify)
			for key in settings:
				for attr in ['ax',*['%s_%s'%('ax',k) for k in AXES],'fig']:
					obj_wrap(attr,key,fig,axes,settings)

		return fig,axes

	def setup(x,y,settings,fig,axes,mplstyle,texify):


		def _setup(settings,_settings):
			_update(settings,_settings)
			return
		def _index(i,N,method='row'):
			
			if method == 'row':
				return [1,N,i+1]
			if method == 'col':
				return [N,1,i+1]				
			elif method == 'grid':
				M = int(np.sqrt(N))+1 if N > 1 else 1
				return [M,M,i+1]
			else:
				return [1,N,i+1]


		_defaults = {None:{}}
		defaults = {'ax':{},'fig':{},'subplot':{},'style':{}}

		if isinstance(settings,str):
			with open(settings,'r') as file:
				settings = json.load(file)
		

		if settings == {}:
			settings.update({None:{}})

		update = y is not None

		if any([key in settings for key in defaults]):
			settings = {key:copy.deepcopy(settings) for key in (y if update and isinstance(y,dict) else [None])}

		if not isinstance(y,dict):
			if not isinstance(y,list):
				y = [y]
			y = {key: y for key in settings}

		if not isinstance(x,dict):
			if not isinstance(x,list):
				x = [x]
			x = {key: x for key in settings}


		for key in settings:
			settings[key].update({k:defaults[k] 
				for k in defaults if k not in settings[key]})

		for i,key in enumerate(y):
			if not isinstance(settings[key]['style'].get('layout'),dict):
				settings[key]['style']['layout'] = {}
			if not all([s in settings[key]['style']['layout'] for s in LAYOUT]):
				settings[key]['style']['layout'].update(dict(zip([*LAYOUT[:DIM],LAYOUT[-1]],_index(i,len(y),'grid'))))
		for key in y:

			_settings = {
				'fig':{
						'set_size_inches':{'w':10,'h':10},
						# 'subplots_adjust':{'wspace':2,'hspace':2},
						# 'tight_layout':{'pad':500,'h_pad':300,'w_pad':300},								   			
						'tight_layout':{},								   			
						'savefig':{'fname':'plot.pdf'},
						'close': {'fig':None},
						**settings[key].get('fig',{})},
				'style':{'layout':{s:settings[key]['style'].get('layout',{}).get(s,1) 
					for s in LAYOUT}}
				}
			if update:						
				plotsettings = settings[key].get('ax',{}).pop('plot',{})				
				_settings.update({'ax':{'plot':[{'x':_x,'y':_y,**(plotsettings if isinstance(plotsettings,dict) else plotsettings[_i])} for _i,(_x,_y) in enumerate(zip(x.get(key,[None]*len(y[key])),y[key]))]},})				
			_setup(settings[key],_settings)	



		for key in settings:
			settings[key].update({k:defaults[k] 
				for k in defaults if k not in settings[key]})

		if fig in [None]:
			fig = {}

		if axes in [None]:
			axes = {}

		for key in settings:
			attr = 'layout'
			layout(key,fig,axes,settings)

			attr = 'style'
			for prop,obj in zip(['mplstyle','texify'],[mplstyle,texify]):
				settings[key][attr][prop] = settings[key][attr].get(prop,obj)

		return settings,fig,axes


	mplstyles = [mplstyle,
				os.path.join(os.path.dirname(os.path.abspath(__file__)),'plot.mplstyle'),
				matplotlib.matplotlib_fname()]

	_mplstyles = [mplstyle,
					os.path.join(os.path.dirname(os.path.abspath(__file__)),'plot_notex.mplstyle'),
					matplotlib.matplotlib_fname()]

	for mplstyle in mplstyles:
		if mplstyle is not None and os.path.isfile(mplstyle):
			break
	for _mplstyle in _mplstyles:
		if _mplstyle is not None and os.path.isfile(_mplstyle):
			break			

	settingss = [settings,
				os.path.join(os.path.dirname(os.path.abspath(__file__)),'plot.json'),
				{}]
	for settings in settingss:
		if ((settings is not None) or (isinstance(settings,str) and os.path.isfile(settings))):
			break

	try:
		fig,axes = context(x,y,settings,fig,axes,mplstyle,texify)
	except:
		rc_params = {'text.usetex': False}
		matplotlib.rcParams.update(rc_params)
		matplotlib.use('pdf') 

		fig,axes = context(x,y,settings,fig,axes,_mplstyle,texify)

	return fig,axes



if __name__ == '__main__':
	if len(sys.argv)<2:
		exit()
	data = sys.argv[1]
	path = sys.argv[2]
	settings = sys.argv[3]
	mplstyle = sys.argv[4]
	Y = sys.argv[5].split(' ')
	X = sys.argv[6].split(' ')



	df = pd.concat([pd.read_csv(d) for d in glob.glob(data)],
					axis=0,ignore_index=True)

	with open(settings,'r') as f:
		_settings = json.load(f)

	settings = {}

	for i,(x,y) in enumerate(zip(X,Y)):
		key = y

		settings[key] = copy.deepcopy(_settings)

		settings[key]['ax']['plot']['x'] = df[x].values if x in df else df.index.values
		settings[key]['ax']['plot']['y'] = df[y].values if y in df else df.index.values
		settings[key]['ax']['set_xlabel'] = {'xlabel':x.capitalize() if x in df else None}
		settings[key]['ax']['set_ylabel'] = {'ylabel':y.capitalize() if y in df else None }
		settings[key]['style']['layout'] = {'ncols':len(Y),'nrows':1,'index':i}
		settings[key]['fig']['savefig'] = {'fname':path,'bbox_inches':'tight'}

	fig,axes = plot(settings=settings,mplstyle=mplstyle) 
