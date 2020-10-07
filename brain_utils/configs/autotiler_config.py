from Algorithmia.errors import AlgorithmException
import logging


class Config_Autotiler:
    '''
    Basically just a class config
    '''

    threshold = 85


    def __init__(self, mode='brain'):
        # avoiding circular import
        from . import class_configs

        if mode == 'brain':
            autotile_configs = class_configs.Config_9_Class()
        elif mode == 'ovarian':
            autotile_configs = class_configs.Config_Clinical_Trials_OV()
        else:
            raise AlgorithmException("Mode {} unknown".format(mode))

        logging.debug("*** - Using {} model ({}) as autotiler - ***".format(mode, autotile_configs.identity))

        # add the class configs properties to this autotiler config object
        self.__dict__.update(autotile_configs.__dict__)
