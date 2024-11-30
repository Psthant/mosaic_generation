import cv2
from mosaic_generator import Resource, Generator, load_resource

resources = Resource(src='database')
# or
# resources = load_resource('file/path/')

generator = Generator(resources, 'london.jpg', (100, ), True)
image = generator.gen()
cv2.imwrite('london-mixed.png', image)
