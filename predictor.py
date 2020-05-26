#!/var/www/html/myproject/myprojectenv/bin/python
import tensorflow as tf
import numpy as np
import cv2
import json
import os
import logging
import time

"""
This version of the predictor aims to take lot photos from a folder and run a prediction 
among each lot photo
"""

class Predictor:

    def logger(self, info):
        logging.basicConfig(filename='/var/www/html/myproject/myprojectenv/debug.log', level=logging.DEBUG, filemode='w')
        logging.debug(info)

    def load_json(self, dir, json_file):
        """
        loads json from pkl_metadata directory
        The [1:] in os.listdir(dir_json)[1:] is to eliminate .DS_Store file

        :return: data which contains parking lot metadata
        """
        file_open = open(dir+json_file, 'r')
        pkl_data = json.load(file_open)
        file_open.close()
        return pkl_data

    def load_parking_spots(self, pkl_metadata, pkl_image):
        """
        This function takes parking spot patches from pkl_image and returns them as an array
        :return: patches
        """
        frames = []
        for x in range(len(pkl_metadata)):
            b = pkl_metadata[x]['bound_yx']

            x1, y1 = int(b[0]), int(b[1])
            x2, y2 = int(b[0] + b[2]), int(b[1] + b[3])

            spot = pkl_image[y1:y2, x1:x2]

            spot = cv2.resize(spot, dsize=(224, 224))
            frames.append(spot)
        parking_spots = np.asarray(frames, dtype='float') / 255.0
        return parking_spots

    def predict_spot(self, model, patches, pkl_data, spot_tags):
        """
        Runs the predictions using all parking space patches and updates the statuses in the json
        associated with that lot

        :param patches: parking space patches
        :param pkl_json: json containing parking lot metadata
        :return: update parking lot json
        """
        for x in range(len(patches)):
            input = np.reshape(np.asarray(patches[x]), newshape=(1,224,224,3))

            prediction = model.predict_classes(input)

            #{'busy': 0, 'free': 1} are the preditcion labels but when written to json busy is 1 and free is 0
            if prediction[0][0] == 0:
                pkl_data['metadata'][spot_tags[x]]['isOccupied'] = int(1)
            elif prediction[0][0] == 1:
                pkl_data['metadata'][spot_tags[x]]['isOccupied'] = int(0)

        return pkl_data

    def write_json(self, file_path, pkl_data):
        """
        Overwrites parking lot json

        :param file_path: path to parking lot json file
        :param pkl_data: updated json information
        :return: 0
        """
        file_write = open(file_path, 'w')
        file_write.write(json.dumps(pkl_data, indent=4))
        file_write.close()
        return 0

    def start_prediction_random_test(self):
        """
        This function goes through every json in folder and updates the spot statuses with random statuses via corresponding
        image in pkl_image folder.

        """
        dir_json = "/var/www/html/myproject/myprojectenv/pkl_metadata/"
        while True:
            for pkl_json in os.listdir(dir_json):
                pkl_data = self.load_json(dir_json,pkl_json)
                pkl_metadata = pkl_data['metadata']

                #Random spot statuses
                rand = np.random.randint(2, size=len(pkl_metadata))
                for i in range(len(pkl_metadata)):
                    pkl_data['metadata'][i]['isOccupied'] = int(rand[i])

                #Overwrite old json
                self.write_json(dir_json+pkl_json, pkl_data)

            time.sleep(10) #Assign random values every 10 seconds


    def start_prediction(self):
        """
        This function goes through every json in folder and updates the spot statuses via corresponding
        image in pkl_image folder.

        """
        dir_json = "/var/www/html/myproject/myprojectenv/pkl_metadata/"
        model = tf.keras.models.load_model('/var/www/html/myproject/myprojectenv/AlexNet4_21_1.h5')  # Loading keras model
        while True:
            for pkl_json in os.listdir(dir_json):

                pkl_data = self.load_json(dir_json,pkl_json)
                for camera in range(len(pkl_data['cameras'])):
                    if pkl_data['cameras'][camera]['last_update'] == 1:

                        #Load data and variables needed
                        #spot_tags should correspond to the indices in pKl_data['metadata']
                        image_path, spot_tags = pkl_data['cameras'][camera]['image_path'], pkl_data['cameras'][camera]['spot_tags']
                        pkl_image = cv2.imread(image_path, cv2.IMREAD_ANYCOLOR)
                        pkl_metadata = [pkl_data['metadata'][x] for x in spot_tags]

                        #Get parking spots and perform predictions
                        parking_spots = self.load_parking_spots(pkl_metadata, pkl_image)
                        pkl_data = self.predict_spot(model, parking_spots, pkl_data, spot_tags)

                        #Overwrite old json
                        pkl_data['cameras'][camera]['last_update'] = 0
                        self.write_json(dir_json+pkl_json, pkl_data)
                time.sleep(0.1) #This can be adjusted

if __name__ == '__main__':
    p = Predictor()
    p.start_prediction()
    #p.start_prediction_random_test()
