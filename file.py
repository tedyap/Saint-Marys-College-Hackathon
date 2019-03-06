
import os
from os import listdir
from shutil import move

def get_data(file_path='Train/'):
	# Get a list of files
	files = listdir(file_path)
	for file in files:
		if file.endswith(".jpg"):
			file_name = file.split(".")[0]
			dir_name = file_name.split("_")[1]
			dst_path = file_path + dir_name # Destination file path
			# Make directories
			if not os.path.exists(dst_path):
				os.makedirs(dst_path)
			move(file_path+file,dst_path)

get_data()