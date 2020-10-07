import cv2
import numpy as np
import logging


class TileUtils:

    @staticmethod
    def add_texts_with_bg(img, texts):
        '''
        Adds each text line by line at the bottom left area of the image

        :param img:
        :param texts:
        :return:
        '''

        font_scale = 2
        thickness = 2
        font = cv2.FONT_HERSHEY_DUPLEX
        # set the rectangle background to white
        rectangle_bgr = (255, 255, 255)
        rectangle_padding = 25

        # set the text start position
        text_offset_x = rectangle_padding - 5
        text_offset_y = img.shape[0] - rectangle_padding

        for txt in texts[::-1]:
            (text_width, text_height) = cv2.getTextSize(txt, font, fontScale=font_scale, thickness=1)[0]

            # make the coords of the box with a small padding of two pixels
            box_coords = (
                (text_offset_x - rectangle_padding - 5, text_offset_y + rectangle_padding),
                (text_offset_x + text_width + rectangle_padding - 5, text_offset_y - text_height - rectangle_padding)
            )
            cv2.rectangle(img, box_coords[0], box_coords[1], rectangle_bgr, cv2.FILLED)
            cv2.putText(img, txt, (text_offset_x, text_offset_y), font, fontScale=font_scale, color=(0, 0, 0),
                        thickness=thickness)

            text_offset_y -= 80

    @staticmethod
    def create_image_vector_for_each_classes(final_classifications, tiles, classes, confs, max_tiles_per_row=3):
        '''
        Makes a vector where each tile belonging to each final classifications (tile will have highest conf)
        and each tile will have an associated description overlaid

        :param final_classifications: list
        :param tiles:
        :param classes:
        :param confs:
        :param max_tiles_per_row: how many tiles to have at most side by side in the tsne figure
        :return:
        '''

        if len(final_classifications) != len(set(final_classifications)):
            raise Exception('Final classifications should be unique')

        UNDEFINED_ANOMALY = 'Undefined Anomaly'

        final_classifications = list(final_classifications)
        if len(final_classifications) > 1 and UNDEFINED_ANOMALY in final_classifications:
            logging.warning(
                f'Final classifications: contains undefined and regular classes ({final_classifications})..removing undefined for image vector')
            final_classifications.remove(UNDEFINED_ANOMALY)

        res = []
        # each displayed tiles' conf score
        chosen_confs = []
        for final_classification in final_classifications:

            texts = []

            if final_classification == UNDEFINED_ANOMALY:
                # find where the largest conf is. we will use the corresponding class
                (_, idx2) = np.unravel_index(np.argmax(confs), confs.shape)
                final_classification = classes[idx2]
                texts.append('(Highest conf)')

            # find the tile with this class as the highest conf
            idx2 = classes.index(final_classification)
            idx1 = np.argmax(confs[:, idx2])
            tile = tiles[idx1]

            chosen_confs.append((final_classification, confs[idx1][idx2]))
            texts.append(final_classification)
            texts.append(f'Conf: {round(confs[idx1][idx2] * 100, 2)}%')

            # add the class and conf score onto the tile
            TileUtils.add_texts_with_bg(tile, texts)
            res.append(tile)

        # create an image vector with these tiles
        res = TileUtils.make_image_vector_using_tiles(res, tiles_per_row=max_tiles_per_row, add_numbering=False)
        return res, chosen_confs

    @staticmethod
    def make_image_vector_using_tiles(tiles, tiles_per_row=3, add_numbering=False):
        '''
        Constructs tiles into a mega image vector.
        Good for viewing tiles that are used in the slide or for whatever use case

        :param tiles:
        :return: numpy image vector matrix
        '''

        if len(tiles) == 0:
            raise Exception("No tiles given. Cannot make image vector")

        tile_size = tiles[0].shape[0]

        num_rows = int(np.ceil(len(tiles) / tiles_per_row))

        # TODO:
        # # if we have at least same number of tiles as max_tiles_per_row, then that is number of columns
        # num_cols = tiles_per_row if len(tiles) >= tiles_per_row else len(tiles)
        # always have x tiles per row. blank pad if necessary
        num_cols = tiles_per_row

        # image holder
        new = np.ones((tile_size * num_rows, tile_size * num_cols, 3), dtype=np.uint8) * 255

        for idx, tile in enumerate(tiles):
            tile = np.array(tile)  # gives error if i dont

            # what slice in the output vector we are in
            curr_row_slice = int(np.floor(idx / tiles_per_row)) * tile_size
            curr_col_slice = idx % (tiles_per_row) * tile_size

            # bit of image editing
            if add_numbering:
                bottom_left_corner_of_text = (25, tile.shape[0] - 50)
                TileUtils.add_text(tile, str(idx + 1), bottom_left_corner_of_text=bottom_left_corner_of_text,
                                   font_scale=8, thickness=10, color=(0, 0, 0))
            TileUtils.add_border(tile, thickness=0.005, color=(0, 0, 0))

            # store image
            new[
            curr_row_slice:curr_row_slice + tile_size, curr_col_slice:curr_col_slice + tile_size, :] = tile

        return new

    @staticmethod
    def add_border(a, thickness=0.05, color=(0, 0, 0)):
        '''

        :param a: the matrix image
        :param thickness: border thickness
        :return:
        '''

        h, w, c = a.shape

        # some coordinates may be part of cropped part of heatmap (recall we do tile size * divide fac). ignore those ones
        if h == 0 or c == 0:
            return

        if c != 3:
            raise Exception('Only RGB images supported')

        pixel_len = min(int(w * thickness), int(h * thickness))

        # for each row in the image
        for j in range(h):
            # if we are in first 5% or last% of rows, we color the whole row
            if j <= pixel_len or j >= w - pixel_len:
                # color entire row
                for i in range(3):
                    a[j, :, i] = color[i]
            else:

                # color the leftmost and rightmost 5% of the row
                for i in range(3):
                    a[j, :pixel_len, i] = color[i]
                    a[j, (w - pixel_len):, i] = color[i]

    @staticmethod
    def add_text(a, text, bottom_left_corner_of_text, color=(0, 0, 0), font_scale=1, thickness=2):
        font = cv2.FONT_HERSHEY_DUPLEX

        cv2.putText(a, text,
                    bottom_left_corner_of_text,
                    font,
                    font_scale,
                    color,
                    thickness)
