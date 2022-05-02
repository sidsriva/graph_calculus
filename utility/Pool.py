#!/usr/bin/env python

import os,sys
import timeit,time,itertools
import numpy as np
from queue import Empty

from threading import Thread as Process
from threading import Event
from queue import Queue

from multiprocessing import Process
from multiprocessing import Event
from multiprocessing import JoinableQueue as Queue



# Worker thread class with Queue of tasks,
# process number, and values to append function 
# returns to
class Worker(Process):
	def __init__(self,tasks,process,values=None,timeout=2):
		Process.__init__(self)
		
		self.tasks = tasks
		self.process = process
		self.values = values

		self.daemon = True
		self.event = Event()
		
		self.start()
		return

	# Run worker loop to check for addition of tasks and to call functions
	def run(self,timeout=2):       
		while not self.event.is_set():
			try:
				func, args, kwargs = self.tasks.get(block=True,timeout=timeout)
				try:
					value = func(*args, **kwargs)
					if self.values is not None:
						self.values.append(value)
				except Exception as e:
					print(e)
					pass
				finally:
					self.tasks.task_done()
			except Empty as e:
				pass
		return

	# Close worker and trigger done event when process is done
	def done(self):
		self.event.set()
		return


# Pool of a processes number of threads
class Pool(object):
	def __init__(self, processes,values=None):

		self.timer()

		self.processes = processes
		self.values = values
		self.workers = []		
		self.tasks = Queue(processes)
		
		self.work()

		return

	# Add workers
	def work(self):
		for process in range(self.processes):
			self.workers.append(Worker(self.tasks,process,self.values))
		return

	# Add to tasks to be completed in Queue
	def put(self, func, *args, **kwargs):
		self.tasks.put((func, args, kwargs))
		return

	# Join tasks in Queue
	def join(self):
		self.tasks.join()
		return

	# Close workers
	def close(self,display=False):
		for worker in self.workers:
			worker.done()
		self.workers = []
		self.timer(display=display)		
		return

	# Time runtime
	def timer(self,display=False,reset=True):
		def settime():
			self.time = timeit.default_timer()
		if not hasattr(self,'time'):
			settime()
		if display:
			print('Time: %0.3e s'%(timeit.default_timer() - self.time))
		if reset:
			settime()
		return 

	# Enter context manager
	def __enter__(self):
		return self

	# Exit context manager and show runtime
	def __exit__(self,type,value,traceback):
		self.join()
		self.close(display=True)
		return


	# Properly close and delete workers
	def __del__(self):
		self.close(display=False)
		return




if __name__ == "__main__":

	def fit(X,y):
		return np.linalg.pinv(X).dot(y)
	def predict(X,y,coef_=None):
		coef_ = fit(X,y) if coef_ is None else coef_
		return X.dot(coef_)
	def loss(y,y_pred,order=2):
		return np.power(np.mean((y-y_pred)**order),1.0/order)
	def func(index,dim,X,y):
		_X = np.delete(X[:,:,dim],index,axis=1)
		_y = y[:,dim]
		coef_ = fit(_X,_y)
		value = loss(_y,predict(_X,_y,coef_))
		return (index,dim,value)
	


	variables = {}
	defaults = {
		'n':{'value':100,'type':int},
		'p':{'value':10,'type':int},
		'm':{'value':10,'type':int},
		'd':{'value':1,'type':int}
		}

	variables.update({k: defaults[k]['value'] for k in defaults})
	variables.update({k: defaults[k]['type'](v) for k,v in zip(variables,sys.argv[1:])})

	n = variables['n']
	p = variables['p']
	m = variables['m']
	d = variables['d']

	X = np.random.rand(n,m,d)
	coef_ = np.random.rand(m)
	y = np.moveaxis(X,[1],[-1]).dot(coef_)	

	N = m
	values = []
	indices = np.arange(N)
	dims = np.arange(d)

	start = timeit.default_timer()
	while (N>0):
		with Pool(p,values=values) as pool:
			for index in indices[:N]:
				for dim in dims:
					pool.put(func,index,dim,X,y)
		N -= 1
		print('N: %d'%(N))
	end = timeit.default_timer()

	print('Time : %0.3e s'%(end-start))
