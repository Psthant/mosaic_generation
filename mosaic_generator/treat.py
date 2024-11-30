import cv2
import numpy as np

def treat(file):
    image = cv2.imread(file, cv2.IMREAD_COLOR)

    dimensions = image.shape[1], image.shape[0]
    min_dim = min(dimensions)
    image = image[
                        int((dimensions[1] - min_dim) / 2): int((dimensions[1] + min_dim) / 2),
                        int((dimensions[0] - min_dim) / 2): int((dimensions[0] + min_dim) / 2)
                    ]

    stack = np.vstack(image)
    col_avg = np.mean(stack, axis=0)
    return image, col_avg