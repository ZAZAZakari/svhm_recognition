# -*- coding: utf-8 -*-
import pickle
import os
import numpy as np
# ==========================================================================================================================
# Save an object to an external pickle file 
# ==========================================================================================================================
# [INPUT]  obj				    	[*]      The object gonna to be dumped to a pickle file, it can be in any data type
#		   fileName               	[String] The filename of the external pickle file
#		   folder 					[String] The directory where the pickle file is goint to save
#===========================================================================================================================
def save(obj, fileName, folder):
	# If the target directory is not exists, create it #
	if not os.path.exists(folder):
		os.makedirs(folder)
	# Join the directory and filename to a valid file path
	fname = os.path.join(folder, '%s.pickle'%(fileName))
	with open(fname, 'wb') as f:
		print('Creating %20s to %35s with shape = %s' % (fileName, fname, np.shape(obj)))
		pickle.dump(obj, f)
# ==========================================================================================================================
# End of code
# ==========================================================================================================================