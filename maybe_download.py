from six.moves import urllib
import os
import sys
import tarfile

# ==========================================================================================================================
# DOWNLOAD A FILE FROM THE GIVEN DATA URL 
# ==========================================================================================================================
# [INPUT]  DEST_DIRECTORY 			[STRING] THE destination where the downloading file is going to save
#		   data_utl               	[String] The download link
# [ACTION] Download the file on the given data url to the destinating folder
#===========================================================================================================================
def downalod(dest_directory, DATA_URL):
    # Check if the destination directory exists, if not, create the folder #
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    # split the file name and file path from the data_url
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    
    # Download the file 
    if not os.path.exists(filepath):
        # define a function to print out message while the download is in process
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')

# ==========================================================================================================================
# Extract compressed file  
# ==========================================================================================================================
# [INPUT]  DEST_DIRECTORY 			[STRING] the path that the compressed file is locating at
#		   filename               	[String] the filename of the file which is going to decompress
# [ACTION] Decompress the file, the extracted folder will be located at the same path
#===========================================================================================================================   
def extract(dest_directory, filename):
    filepath = os.path.join(dest_directory, filename)
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)
    
# ==========================================================================================================================
# End of code
# ==========================================================================================================================