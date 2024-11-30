from mosaic_generator import Resource, Generator, load_resource

resources = Resource(src='database-cats')
resources.save('resources')

r = load_resource('resources')
