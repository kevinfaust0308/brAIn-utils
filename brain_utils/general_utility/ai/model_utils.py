import numpy as np
import tensorflow.keras.backend as K
from fuzzywuzzy import process, fuzz
import logging


class ModelUtils:


    @staticmethod
    def remove_class(pred, classes, remove=['blank', 'marker'], redist_conf=True, **kwargs):
        '''
        Removes classes from the pred and redistributes the removed preds so that the pred still sums to 1.
        Can pass in additional lists to remove value at the index corresponding to the removed pred

        Does not require for the items in 'remove' to be in the classes exactly BUT must be similar

        ie pred=[0.2,0.6,0.2], classes=['blank', ..., ...], kwargs={'a': [1,2,3]}
        then this method returns {'pred': [0.7,0.3], 'classes': [..., ...], 'a': [2, 3]}

        :param pred:
        :param classes:
        :param remove:
        :param redist_conf:
        :return:
        '''

        pred, classes = np.array(pred), list(classes)

        for r in remove:
            # find the exact name of 'remove' within classes list
            name, confidence = process.extractOne(r, classes, scorer=fuzz.partial_ratio)
            if confidence < 85:
                logging.warning("Failed to remove {}. Seems to not be present in given class list".format(r))
                # we will return the arguments with nothing removed
            else:
                # delete
                idx = classes.index(name)
                classes.pop(idx)
                pred = np.delete(pred, idx)
                # delete element at index in all kwargs values. on the copy
                for k, v in kwargs.items():
                    v = v.copy()
                    v.pop(idx)
                    kwargs[k] = v

                if redist_conf:
                    # redistribute
                    ### METHOD 1: uniformly distribute
                    # redistribute_amnt = (1 - pred.sum()) / len(pred)
                    # pred += redistribute_amnt
                    ### METHOD 2: keep ratios between preds ie [50% (blank), 10%, 40%] ->  [20%, 80%] (still x4)
                    pred = pred / pred.sum()

        result = {
            'pred': pred,
            'classes': classes
        }
        result.update(kwargs)
        return result


    @staticmethod
    def get_layer_datas(model, imgs, layers):
        '''
        Returns image data after the specified layer

        :param model: keras model
        :param imgs: list of images
        :param layers: list of string layer names or keras layers within the model
        :return: list of numpy arrays; each element corresponds to each layer output
        '''

        if len(layers) == 0:
            raise Exception('No layers specified')

        imgs = ModelUtils.prepare_images(imgs)

        layer_outputs = [model.get_layer(l).output if type(l) == str else l.output for l in layers]
        get_output = K.function(model.layers[0].input, layer_outputs)

        return get_output(imgs)


    @staticmethod
    def prepare_images(imgs):
        '''
        Returns an array of prepared images for model use

        :param img: array of images
        :return: array of images
        '''

        return np.divide(imgs, 255)
