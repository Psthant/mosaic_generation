import cv2
import random
import numpy as np
from .resource import Resource, load_resource

class Generator:
    def __init__(self, resource_obj: Resource, path: str, dimensions: tuple, square_dimensions: bool =True):
        self.resource = resource_obj.data
        self.path = path
        self.dimensions = dimensions
        self.sqaure_dimensions = square_dimensions

    def gen(self) -> np.ndarray:
        """
        Generate a mosaic image from the dataset
        """
        image = cv2.imread(self.path)
        if self.sqaure_dimensions:
            self.dimensions = self.__square_dimensions(image, self.dimensions[0])
        parts, row_height, col_width = self.__pixellate(image, rows=self.dimensions[0], cols=self.dimensions[1])
        image = self.__reconstruct(parts, row_height, col_width)
        return image

    def __square_dimensions(self, image: np.ndarray, n: int) -> tuple[int]:
        dimensions = image.shape[1], image.shape[0]
        row_height = np.floor(dimensions[0] / n)
        col = dimensions[1] // row_height
        return n, int(col)
    
    def __pixellate(self, image: np.ndarray, rows: int, cols: int) -> list[np.ndarray] | int | int:
        dimensions = image.shape[1], image.shape[0]
        row_height = int(np.floor(dimensions[0] / rows))
        col_width = int(np.floor(dimensions[1] / cols))
        image = cv2.resize(image, (row_height * rows, col_width * cols))

        parts = []
        for j in range(cols):
            for i in range(rows):
                part = image[j * col_width: (j+1) * col_width, i * row_height: (i+1) * row_height]
                parts.append(part)
        return parts, row_height, col_width
    
    def __find_similar(self, input_col: tuple, euclidean_distance: int) -> list[np.ndarray]:
        similarity_rating = []
        for image, colour_sample in self.resource:
            a = np.square(colour_sample - input_col)
            r = np.sqrt(sum(a))
            similarity_rating.append(r)
        min_value = min(similarity_rating)
        images = []
        for i, rating in enumerate(similarity_rating):
            if abs(rating - min_value) <= euclidean_distance:
                images.append(self.resource[i][0])
        return random.choice(images)
    
    def __reconstruct(self, parts, row_height, col_width, euclidean_distance: int = 10):
        rows, cols = self.dimensions
        row_concat = []
        for i in range(cols):
            col_concat = []
            init_array = np.empty((col_width, row_height, 3))
            for j in range(rows):
                image = parts[i * rows + j]
                stack = np.vstack(image)
                col_avg = np.mean(stack, axis=0)
                image = self.__find_similar(col_avg, euclidean_distance)
                image = cv2.resize(image, (row_height, col_width))

                col_concat.append(image)
            row_concat.append(cv2.hconcat(col_concat))
        image = cv2.vconcat(row_concat)
        return image