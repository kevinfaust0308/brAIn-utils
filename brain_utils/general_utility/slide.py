import pathlib
from collections import namedtuple
from PIL import Image
import warnings

Image.MAX_IMAGE_PIXELS = 100000000000


class Slide:
    '''
    A slide object
    '''


    def __init__(self, path, img_requirements=None, stain_type='Unknown'):
        '''
        Creates a slide object with all possible data of the slide extracted

        :param path:
        :param img_requirements: dictionary of required svs configurations
        '''

        self.path = path
        self.name = pathlib.Path(path).stem
        self.image_type = pathlib.Path(path).suffix
        self.stain_type = stain_type

        i = Image.open(path)
        self.width, self.height = i.width, i.height
        self.image = i

        Coordinate = namedtuple('Coordinate', 'x y')
        self.start_coordinate = Coordinate(0, 0)

        # get svs data if its an svs path
        curr_slide_data = self._extract_data(path)
        self.date_scanned = curr_slide_data['date_scanned']
        self.time_scanned = curr_slide_data['time_scanned']
        self.compression = curr_slide_data['compression']
        self.mpp = curr_slide_data['mpp']
        self.apparent_magnification = curr_slide_data['apparent_magnification']  # only here while in process of removal


    def crop(self, coordinates):
        '''
        Updates internal slide properties so that we will only use a section of the slide

        :param coordinates: use only a section of the slide (top_left_x, top_left_y, bot_right_x, bot_right_y)
        :return:
        '''
        Coordinate = namedtuple('Coordinate', 'x y')
        self.start_coordinate = Coordinate(coordinates[0], coordinates[1])
        self.width = coordinates[2] - coordinates[0]
        self.height = coordinates[3] - coordinates[1]


    def get_thumbnail(self, wh_dims):
        '''

        :param dims: dimensions of returned thumbnail (width, height)
        :return:
        '''

        warnings.warn('Please use thumbnail generated from heatmap for faster results')
        return Image.open(self.path).resize(wh_dims)


    def is_valid_img_file(self, img_requirements):
        '''
        Returns true if the slide is a svs that satisfies the image requirements.
        Image requirements can specify None if a specific property is unrestricted
        If image is a jpg, trivially returns true


        :param img_requirements: dictionary of required svs configurations
        :return: boolean
        '''

        return True


    def _extract_data(self, slide_path):
        '''
        Extracts useful metadata from the svs

        :param slide_path:
        :return:
        '''

        return {
            'date_scanned': None,
            'time_scanned': None,
            'compression': None,
            'mpp': None,
            'apparent_magnification': None
        }
