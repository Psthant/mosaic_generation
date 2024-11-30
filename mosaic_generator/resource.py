import os
import numpy as np
import datetime
import cv2
from .treat import treat

class Resource:
    def __init__(self, src='', data=[]):
        if src and data:
            raise ValueError('src and data arguments cannot be passed simultaneously')
        if src:
            self.data = []
            for file in os.listdir(f'{src}/'):
                image_vector, color_vector = treat(f'{src}/{file}')
                self.data.append((image_vector, color_vector))
        else:
            self.data = data

    def __get_name(self):
        dt = datetime.datetime.now()
        return f'{dt.year}{dt.month}{dt.day}-{dt.microsecond}'

    def save(self, dst):
        for image_vector, color_vector in self.data:
            data_vector = np.append(image_vector, color_vector, axis=0)
            color_vector = np.tile(color_vector, (image_vector.shape[0], 1)).reshape(1, image_vector.shape[0], 3)
            cv2.imwrite(f'{dst}/{self.__get_name()}.png', data_vector)
        print(f'Saved to {os.getcwd()}\\{dst}\\')

def load_resource(src: str) -> Resource:
    """
    Return a `Resource` dataset
    """
    data = []
    for file in os.listdir(f'{src}/'):
        image = cv2.imread(f'{src}/{file}')
        colour = image[-1, :, :]
        colour = np.ravel(colour)[:3]
        image = image[:-1, :, :]
        data.append((image, colour))
    return Resource(data=data)