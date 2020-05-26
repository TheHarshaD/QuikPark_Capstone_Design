import os.path
import json

def send_pkl_data(pkl_name):
	""" 
	Handles GET request from IOS app

	Displays the contents of a given parking lots json to webpage where IOS app
	then grabs that information to be used
	"""
	file_path = '/var/www/html/myproject/myprojectenv/pkl_metadata/'+pkl_name+'.json'
	if os.path.isfile(file_path):
		file_open = open(file_path, 'r')
		pkl_data = json.load(file_open) 
		file_open.close()
		return pkl_data
	else:
		return 'Error: file not found'
