#!/usr/bin/env python

# Import python modules
import sys,os,glob,copy,itertools
from natsort import natsorted
import numpy as np
import pandas as pd
# import numexpr as ne

# Global Variables
CLUSTER = 1
PATH = '../'

DELIMITER='__'
MAX_PROCESSES = 7
PARALLEL = 0

# ne.set_vml_num_threads(MAX_PROCESSES)


# Import user modules
# export mechanoChemML_SRC_DIR=/home/matt/files/um/code/mechanochem-ml-code
environs = {'mechanoChemML_SRC_DIR':[None,'utility']}
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
			directory = os.path.join(directory,path)
		sys.path.append(directory)

from utility.graph_settings import set_settings,get_settings,permute_settings

from utility.graph_functions import structure,terms,save,analysis

from utility.graph_models import model

from utility.graph_fit import fit

from utility.graph_plot import plotter

from utility.texify import Texify,scinotation

from utility.dictionary import _set,_get,_pop,_has,_update,_permute

from utility.load_dump import setup,load,dump,path_split,path_join
		



# Logging
import logging,logging.handlers
log = 'info'

rootlogger = logging.getLogger()
rootlogger.setLevel(getattr(logging,log.upper()))
stdlogger = logging.StreamHandler(sys.stdout)
stdlogger.setLevel(getattr(logging,log.upper()))
rootlogger.addHandler(stdlogger)	


logger = logging.getLogger(__name__)
logger.setLevel(getattr(logging,log.upper()))




def main(data={},metadata={},settings={}):
	""" 
	Main program for graph theory library

	Args:
		data (dict): dictionary of {key:df} string keys and Pandas Dataframe datasets
		metadata (dict): dictionary of {key:{}} string keys and dictionary metadata about datasets
		settings (dict): settings in JSON format for library execution
	"""

	# Set Settings
	set_settings(settings,
				path=path_join(settings.get('sys',{}).get('src',{}).get('dump',''),
						  settings.get('sys',{}).get('settings')) if (
					(isinstance(settings.get('sys',{}).get('settings'),str)) and (
					not (settings.get('sys',{}).get('settings').startswith(settings.get('sys',{}).get('src',{}).get('dump',''))))) else (
					settings.get('sys',{}).get('settings')),
				_dump=True,_copy=False)


	# Import datasets
	if any([k in [None] for k in [data,metadata]]):
		data = {}
		metadata = {}
	if any([k in [{}] for k in [data,metadata]]):
		setup(data=data,metadata=metadata,
			  files=settings['sys']['files']['files'],
			  directories__load=settings['sys']['directories']['directories']['load'],# if ((not settings['boolean']['load']) or (not settings['boolean']['dump'])) else settings['sys']['directories']['directories']['dump'],
			  directories__dump=settings['sys']['directories']['directories']['dump'],
			  metafile=settings['sys']['files']['metadata'],
			  wr=settings['sys']['read']['files'],
			  flatten_exceptions=[],
			  **settings['sys']['kwargs']['load'])

	verbose = settings['sys']['verbose'] 
	models = {}

	# Set logger and texifying
	if settings['boolean']['log']:
		filelogger = logging.handlers.RotatingFileHandler(path_join(
				settings['sys']['directories']['cwd']['dump'],
				settings['sys']['files']['log'],
				ext=settings['sys']['ext']['log']))
		fileloggerformatter = logging.Formatter(
			fmt='%(asctime)s: %(message)s',
			datefmt='%Y-%m-%d %H:%M:%S')
		filelogger.setFormatter(fileloggerformatter)
		filelogger.setLevel(getattr(logging,log.upper()))
		if len(rootlogger.handlers) == 2:
			rootlogger.removeHandler(rootlogger.handlers[-1])
		rootlogger.addHandler(filelogger)
		


	logger.log(verbose,'Start')
	logger.log(verbose,'Set Settings')

	# Show Directories
	logger.log(verbose,'Imported Data: %s'%(settings['sys']['identity']) if len(data)>0 else "NO DATA")
	logger.log(verbose,'Datasets: %s'%('\n\t'.join(['',*[r'%s: %r'%(key,data[key].shape) for key in data]])))
	logger.log(verbose,'Load paths: %s'%('\n\t'.join(['',*settings['sys']['directories']['directories']['load']])))
	logger.log(verbose,'Dump paths: %s'%('\n\t'.join(['',*settings['sys']['directories']['directories']['dump']])))


	# Texify operation
	if settings['boolean']['texify']:
		tex = Texify(**settings['texify'])
		texify = tex.texify
	else:
		tex = None
		texify = None


	logger.log(verbose,'Setup Texify')

	# Define Structure of graph and perform pre-processing on datasets
	if settings['boolean']['structure']:
		structure(data,metadata,settings,verbose=settings['structure']['verbose'])
	logger.log(verbose,'Defined Structure')


	# Calculate terms
	if settings['boolean']['terms']:
		terms(data,metadata,settings,verbose=settings['terms']['verbose'],texify=texify)
	logger.log(verbose,'Calculated Operators')



	# Save Data
	if settings['boolean']['dump']:
		save(settings,paths={key: metadata[key]['directory']['dump'] for key in data},data=data,metadata=metadata)
	logger.log(verbose,'Saved Data')  


	# Calculate Model
	if settings['boolean']['model']:
		model(data,metadata,settings,models,verbose=settings['model']['verbose'])
	logger.log(verbose,'Setup Model')


	# Save Data
	if settings['boolean']['dump']:
		save(settings,paths={key: metadata[key]['directory']['dump'] for key in data},data=data,metadata=metadata)
	logger.log(verbose,'Saved Data') 


	# Fit Data     
	if settings['boolean']['fit']:
		for label in models:	
			fit(data,metadata,
				{key: metadata[key]['rhs_lhs'].get(label,{}).get('rhs') for key in data},
				{key: metadata[key]['rhs_lhs'].get(label,{}).get('lhs') for key in data},
				label,
				settings['fit']['info'],
				models[label],
				settings['fit']['estimator'],
				{
					**settings['fit']['kwargs'],
					**settings['model'],
					**{'modelparams':settings['analysis']}
					},				
				verbose=settings['fit']['verbose']
				)
	logger.log(verbose,'Fit Data')   


	# Save Data
	if settings['boolean']['dump']:
		save(settings,paths={key: metadata[key]['directory']['dump'] for key in data},data=data,metadata=metadata)
	logger.log(verbose,'Saved Data')  

	# Analyse Fit Results and Save Texified Model
	if settings['boolean']['analysis']:
		analysis(data,metadata,settings,models,texify,verbose=settings['analysis']['verbose'])
	logger.log(verbose,'Analysed Results')  

	# Plot Fit Results
	if settings['boolean']['plot']:
		plotter(data,metadata,settings,models,texify,verbose=settings['plot']['verbose'])
	logger.log(verbose,'Plotted Data')    

	logger.log(verbose,'Done\n')
	 


	return

