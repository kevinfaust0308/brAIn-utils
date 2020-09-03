import numpy as np
import cv2
import time


class TileExtractor:
    DEFAULT_MIN_NON_BLANK_AMT = 0.1


    def __init__(self, slide, tile_size=1024, desired_tile_mpp=0.5040):
        '''
        Creates a tile extractor object for the given slide

        Default MPP used is for 20x magnification. Regardless of the the MPP of the slide given is, the tiles produced
        will be of the `desired_tile_mpp`

        If desired MPP and slide MPP match, then this serves as a raw tile extractor.
        If they don't match:
            if slide MPP is smaller (larger magnification), takes a larger tile size and resizes the tile down
            if slide MPP is larger (smaller magnification), takes a smaller tile size and resizes the tile up

        :param slide: slide object
        :param tile_size:
        :param desired_tile_mpp: the mpp of tiles that the tile extractor returns
        '''

        self.slide = slide
        self.original_tile_size = tile_size
        self.desired_tile_mpp = desired_tile_mpp

        # resize tile size if required. if the slide's mpp has too high precision, round down to avoid numerical issues
        factor = desired_tile_mpp / round(slide.mpp, 3) if slide.mpp else 1
        modified_tile_size = int(tile_size * factor)
        self.tile_size_resize_factor = factor
        self.modified_tile_size = modified_tile_size

        # 'Crop' leftover from right and bottom
        self.trimmed_width = slide.width - (slide.width % modified_tile_size)
        self.trimmed_height = slide.height - (slide.height % modified_tile_size)
        self.chn = 3


    @staticmethod
    def amount_blank(tile):
        '''
        Returns percentage of how many pixels are "blank" within the tile

        :param tile: BGR numpy array
        :return:
        '''

        # in rgb form, when pixels are all equal, it is white or black or grey. get all the corresponding pixels whose
        # values are all similar to each other (using std dev)
        return np.sum(np.std(tile, axis=2) < 3) / (tile.shape[0] * tile.shape[1])


    def iterate_tiles(self, min_non_blank_amt=0.0, batch_size=4, print_time=True):
        '''
        A generator that iterates over all the tiles within the supplied slide

        :param min_non_blank_amt: tile must have at least this percentage of its pixels "non-blank" ie if the value
        is 0.6, means the tile must have 60%+ of its pixels non-blank
        :param batch_size: get x tiles at once
        :param print_time: for printing out how many tiles/how many to go
        :return: dict containing array of tiles and coordinates
        '''

        if not (0 <= min_non_blank_amt <= 1):
            raise Exception("Minimum non-blank amount must be a percentage between 0.0 and 1.0")

        if batch_size < 1:
            raise Exception('Batch size must be at least 1')

        # initialization
        x = y = 0
        tile_size = self.modified_tile_size

        # For timing and count tiles
        cols = self.trimmed_width / tile_size
        rows = self.trimmed_height / tile_size
        tot_tiles = cols * rows
        start_time = time.clock()

        # buffer for our batches. will keep updating this each yield
        tiles_buffer = np.zeros((batch_size, tile_size, tile_size, self.chn), dtype=np.uint8)
        coordinates_buffer = np.zeros((batch_size, 4), dtype=int)
        buffer_i = 0

        # break out when top left coordinate of next tile is the bottom of image
        while y != self.trimmed_height:

            # Get current sub-image
            tile = np.array(self.slide.image.crop((x, y, x + tile_size, y + tile_size)))[:, :, 2::-1]

            r = 1 / self.tile_size_resize_factor
            if r != 1:
                tile = cv2.resize(tile, (0, 0), fx=r, fy=r)

            # only yield if under maximum blank allowance
            if TileExtractor.amount_blank(tile) <= (1 - min_non_blank_amt):
                top_left_x, top_left_y = int(x * r), int(y * r)
                bot_right_x, bot_right_y = int((x + tile_size) * r), int((y + tile_size) * r)
                coordinate = (top_left_x, top_left_y, bot_right_x, bot_right_y)

                tiles_buffer[buffer_i] = tile
                coordinates_buffer[buffer_i] = coordinate
                buffer_i += 1

                if buffer_i == batch_size:
                    buffer_i = 0
                    yield {'tiles': tiles_buffer.copy(), 'coordinates': coordinates_buffer.copy()}

            # move onto next spot
            x += tile_size
            if x == self.trimmed_width:
                x = 0
                y += tile_size

                if print_time:
                    print("{:0.2f}% ({}/{} tiles) in {:0.2f}s".format(
                        (y / tile_size) / rows * 100,  # percent of rows complete
                        (y / tile_size) * cols,  # number of rows complete * tiles per row
                        tot_tiles,
                        time.clock() - start_time))

        # may have leftover tiles
        if buffer_i > 0:
            yield {'tiles': tiles_buffer[:buffer_i, :, :, :], 'coordinates': coordinates_buffer[:buffer_i, :]}


    def iterate_tiles_with_lesion_conf(self, model, non_lesion_indices, min_non_blank_amt=0.0, batch_size=4,
                                       print_time=True):
        '''
        A generator that iterates over all the tiles within the supplied slide along with the lesional score
        '''

        from .model_utils import ModelUtils

        # generator for extracting tiles
        extractor_gen = self.iterate_tiles(
            min_non_blank_amt=min_non_blank_amt, batch_size=batch_size, print_time=print_time)

        for res in extractor_gen:
            tile_batch, coordinate_batch = res['tiles'], res['coordinates']

            # get predictions for each tile
            # preds = ModelUtils.get_conf_scoreJAJA(model, tile_batch)
            preds = model.predict_on_batch(ModelUtils.prepare_images(tile_batch))
            # depends on tensorflow
            try:
                lesion_confs = (1 - np.sum(preds.numpy()[:, non_lesion_indices], axis=1))
            except:
                lesion_confs = (1 - np.sum(preds[:, non_lesion_indices], axis=1))

            yield {
                'tiles': tile_batch,
                'coordinates': coordinate_batch,
                'lesion_confs': lesion_confs,
            }
