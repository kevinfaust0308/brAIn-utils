import numpy as np
import cv2

from .tile_image_utils import TileUtils


class ImageCreator:

    def __init__(self, height, width, scale_factor=1, channels=3):
        '''

        Creates an image of the specified size except scaled down by an optional factor amount

        :param height:
        :param width:
        :param scale_factor: scale the created image down by a factor of this amount
        :param channels:
        '''

        # this matrix will always have the specified dtype
        self.image = np.ones((int(height / scale_factor), int(width / scale_factor), channels), dtype=np.uint8) * 255
        self.scale_factor = scale_factor

    def _get_scaled_coordinate(self, coordinate):
        return tuple(int(c / self.scale_factor) for c in coordinate)

    def add_tile(self, tile, coordinate):
        '''

        :param tile: a 0-255 valued matrix
        :param coordinate: tuple containing top left and bottom right coordinate of image
        :return:
        '''

        # adjust coordinates depending on if we want to scale our image
        x1_adj, y1_adj, x2_adj, y2_adj = self._get_scaled_coordinate(coordinate)

        # Put sub-image into correct spot of matrix (recreating image) by resizing tile if needed to fit within the spot
        self.image[y1_adj:y2_adj, x1_adj:x2_adj, :] = cv2.resize(tile, (x2_adj - x1_adj, y2_adj - y1_adj))

    def add_borders(self, coordinates, color=(0, 255, 0), add_big_text=True):
        '''
        Adds colored borders onto the image at the coordinates. Default is bright green

        :param coordinates: tuple containing top left and bottom right coordinate of image
        :param color: BGR tuple
        :param add_big_text: number the tiles
        :return:
        '''

        if add_big_text:
            # this is used when there's not enough lesional tiles and we put all those lesional tiles on the tile display.
            # we want to show where in the heatmap/image each extracted lesional tile is located.
            # since we numbered the lesional tiles in the tile display in the order of coordinates, we do the same
            # here to have matching numbering.
            add_big_text = len(coordinates) < 15

        for idx, coordinate in enumerate(coordinates):

            # adjust coordinates depending on if we want to scale our image
            x1_adj, y1_adj, x2_adj, y2_adj = self._get_scaled_coordinate(coordinate)

            curr_slice = self.image[y1_adj:y2_adj, x1_adj:x2_adj, :]

            TileUtils.add_border(curr_slice, thickness=0.1, color=color)
            # curr_slice = cv2.copyMakeBorder(curr_slice, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(0,1,0))

            if add_big_text:
                # big black text
                font_scale = 20
                text_color = (0, 0, 0)
                thickness = 40

                # in order for the text to be accurate irrespective of scale factor, we scale to original tile size
                # and put text there before resizing back to previous.
                # TODO: TUNED FOR SIZE 1024x1024. very slight loss of quality but the pros outweight cons
                curr_slice = cv2.resize(curr_slice,
                                        (curr_slice.shape[0] * self.scale_factor,
                                         curr_slice.shape[1] * self.scale_factor))

                bottom_left_corner_of_text = (40, curr_slice.shape[0] - 150)

                TileUtils.add_text(curr_slice, str(idx + 1), bottom_left_corner_of_text=bottom_left_corner_of_text,
                                   font_scale=font_scale, thickness=thickness, color=text_color)
                curr_slice = cv2.resize(curr_slice,
                                        (curr_slice.shape[0] // self.scale_factor,
                                         curr_slice.shape[1] // self.scale_factor))
                # import matplotlib.pyplot as plt
                # plt.imshow(curr_slice)
                # plt.show()

                self.image[y1_adj:y2_adj, x1_adj:x2_adj, :] = curr_slice
