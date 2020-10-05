import numpy as np
import os
import zipfile
from brain_utils.general_utility import unique_colors
from Algorithmia.errors import AlgorithmException
import seaborn as sns

from . import settings


class io:

    def __init__(self, config):
        '''
        Creates all the io aspects for the config object
        NOTE: Configured specifically for algorithmia

        :param config:
        '''

        data_collection_dir_uri = 'data://.my/{}'.format(config.identity)

        self.config = config
        self.save_dir = '/tmp'

        ### store model path
        model_file_name = config.identity + '_VGG19'
        if hasattr(config, 'deprecated_model_type') and config.deprecated_model_type:
            model_ext = '.h5'
        else:
            model_ext = '.zip'
        model_file_name_with_ext = model_file_name + model_ext

        model_uri = data_collection_dir_uri + '/' + model_file_name_with_ext
        self.model_path = settings.client.file(model_uri).getFile().name

        if not (hasattr(config, 'deprecated_model_type') and config.deprecated_model_type):
            # model is stored as zip file. need to extract
            out_dir = extract_model(model_uri)
            self.model_path = out_dir + '/' + model_file_name

        ### store optional confusion matrix path if present
        confusion_matrix_uri = data_collection_dir_uri + '/' + model_file_name + '_test_confusion_matrix.jpg'
        if settings.client.file(confusion_matrix_uri).exists():
            self.confusion_matrix_path = settings.client.file(confusion_matrix_uri).getFile().name
        else:
            self.confusion_matrix_path = None

        ### store optional fv related paths if present
        gap_data_uri = data_collection_dir_uri + '/' + 'GAP_output_data_pre_scale.npy'
        data_labels_uri = data_collection_dir_uri + '/' + 'data_labels.npy'
        if settings.client.file(gap_data_uri).exists() and settings.client.file(data_labels_uri).exists():
            self.fv_data_path = settings.client.file(gap_data_uri).getFile().name
            self.data_labels_path = settings.client.file(data_labels_uri).getFile().name


### NOTE: brain autotiler
class Config_9_Class:

    def __init__(self):
        self.type = 'brain'
        self.folder_name = '9_classes'

        self.mpp = 0.5040
        self.tile_size = 1024
        self.model_type = 'VGG19'
        self.deprecated_model_type = True

        self.lesion_color = 'brown'
        self.non_lesion_colormaps = {
            'a_blank': 'white', 'b_white': 'green', 'c_grey': 'grey',
            'm_necrosis': 'blue', 'n_blood': 'red', 'o_surgical': 'orange',
            'r_dura': 'purple', 's_cerebellum': 'yellow'
        }

        self.classes = [
            'a_blank',
            'b_white',
            'c_grey',
            'l_lesion',
            'm_necrosis',
            'n_blood',
            'o_surgical',
            'r_dura',
            's_cerebellum'
        ]

        setup_config(self)


class Config_42_Class_Custom:

    def __init__(self):
        self.type = 'brain'
        self.folder_name = '42_classes_custom'

        self.mpp = 0.5040
        self.tile_size = 1024
        self.model_type = 'VGG19'
        self.deprecated_model_type = True

        self.lesion_color = 'brown'
        self.non_lesion_colormaps = {
            'Salivary Gland': 'green',
            'Acute Hematoma': 'red',
            'Crush Artifact': 'yellow',
            'Blank': 'white',
            'White': 'blue',
            'Gray': 'grey',
            'Muscle': 'purple',
            'Bone Marrow': 'cyan',
            'Necrosis': 'magenta',
            'Surgical': 'pink',
            'Spinal Disc': 'navy',
            'Dura': 'beige',
            'Cerebellum': 'orange',
            'Lymph Node': 'black'
        }

        self.classes = [
            'Abscess',
            'Salivary Gland',
            'Acute Hematoma',
            'Choroid Plexus Papilloma',
            'Crush Artifact',
            'Lymphoma',
            'Blank',
            'White',
            'Chondrosarcoma',
            'Chronic Hematoma',
            'Clear Cell Renal Cell Carcinoma',
            'Gray',
            'DNET',
            'Chordoma',
            'Muscle',
            'Ganglioglioma',
            'Glioma, WHO 2-4',
            'Psammomatous Meningioma',
            'Bone Marrow',
            'Meningioma',
            'Metastatic Carcinoma',
            'Necrosis',
            'Neurofibroma',
            'Surgical',
            'Paraganglioma',
            'Plasmacytoma',
            'Spinal Disc',
            'Schwannoma',
            'Dura',
            'Sarcoma',
            'Small Blue Cell Tumor',
            'Cerebellum',
            'Hemangioblastoma',
            'Glioma, WHO 1',
            'Myxopapillary',
            'Epidermoid Cyst',
            'Malignant Melanoma',
            'Radiation Necrosis',
            'Lymph Node'
        ]

        setup_config(self)


class Config_Clinical_Trials_OV:

    def __init__(self):
        self.type = 'ovarian'
        self.folder_name = 'clinical_trials_ov_3'

        self.mpp = 0.5040
        self.tile_size = 1024
        self.model_type = 'VGG19'
        # algorithmia giving weird errors when i dont use .h5 format
        self.deprecated_model_type = True

        self.lesion_color = 'brown'
        self.non_lesion_colormaps = {
            'Blank': 'white', 'Marker': 'white', 'Adipose Tissue': 'blue',
            'Necrosis': 'red', 'Normal Ovarian Stroma': 'grey',
            'Normal Glandular Tissue': 'cyan', 'Inflammatory Cells': 'yellow'
        }

        self.classes = [
            'Blank',
            'Normal Glandular Tissue',
            'Adipose Tissue',
            'Necrosis',
            'Normal Ovarian Stroma',
            'Ovarian Serous Adenocarcinoma',
            'Marker',
            'Inflammatory Cells'
        ]

        setup_config(self)


class Config_80_Class_Blank_Filter:

    def __init__(self):
        self.type = 'brain'
        self.folder_name = '80_class_blank_filter'

        self.mpp = 0.5040
        self.tile_size = 1024
        self.model_type = 'VGG19'
        self.deprecated_model_type = True

        self.lesion_color = 'brown'
        self.non_lesion_colormaps = {
            'Salivary Gland': 'green',
            'Acute Hematoma': 'red',
            'Crush Artifact': 'yellow',
            'Blank': 'white',
            'White': 'blue',
            'Gray': 'grey',
            'Muscle': 'purple',
            'Bone Marrow': 'cyan',
            'Necrosis': 'magenta',
            'Surgical': 'pink',
            'Spinal Disc': 'brown',
            'Dura': 'beige',
            'Cerebellum': 'orange',
            'Lymph Node': 'black'
        }

        self.classes = [
            'Abscess',
            'Salivary Gland',
            'Acute Hematoma',
            'Squamous Cell Carcinoma',
            'Choroid Plexus Papilloma',
            'Papillary Craniopharyngioma',
            'Adamantinomatous Craniopharyngioma',
            'Crush Artifact',
            'Lymphoma',
            'Atypical Meningioma',
            'Blank',
            'White',
            'Chondrosarcoma',
            'Chronic Hematoma',
            'Clear Cell Renal Cell Carcinoma',
            'Gray',
            'DNET',
            'Chordoma',
            'Ependymoma',
            'Muscle',
            'Adipose',
            'Ganglioglioma',
            'PilocyticAstrocytoma',
            'Diffuse Astrocytoma, IDH-mut',
            'Diffuse Astrocytoma, IDH-WT',
            'Anaplastic Astrocytoma, IDH-mut',
            'Anaplastic Astrocytoma, IDH-WT',
            'Diffuse Oligodendroglioma, IDH-mut, 1p19-codel',
            'Anaplastic Oligodendroglioma, IDH-mut, 1p19-codel',
            'Glioblastoma, IDH-wt',
            'Glioblastoma, IDH-mut',
            'Anaplastic Meningioma',
            'Angiomatous Meningioma',
            'Chordoid Meningioma',
            'ClearCellMeningioma',
            'Fibrous Meningioma',
            'Meningothelial Meningioma',
            'Microcystic Meningioma',
            'Papillary Meningioma',
            'Psammomatous Meningioma',
            'Secretory Meningioma',
            'Transitional Meningioma',
            'Bone Marrow',
            'Hemangiopericytoma',
            'Liposarcoma - Grade III',
            'Liposarcoma - High Grade',
            'Liposarcoma - Myxoid',
            'Lung Adenocarcinoma',
            'Medulloblastoma',
            'Breast Adenocarcinoma',
            'Colorectal Adenocarcinoma',
            'Prostate Adenocarcinoma',
            'MPNST',
            'Necrosis',
            'Neurocytoma',
            'Neurofibroma',
            'Surgical',
            'Paraganglioma',
            'Pituitary Adenoma',
            'Plasmacytoma',
            'Spinal Disc',
            'Schwannoma',
            'Dura',
            'Sarcoma - Ewing',
            'Sarcoma - High Grade',
            'Small Cell Carcinoma',
            'Cerebellum',
            'Hemangioblastoma',
            'Subependymoma',
            'Myxopapillary Ependymoma',
            'Epidermoid Cyst',
            'Malignant Melanoma',
            'Radiation Necrosis',
            'Lymph Node'
        ]

        setup_config(self)

    class Config_MIB:

        def __init__(self):
            self.type = 'brain'
            self.folder_name = 'mib'

            self.mpp = 0.5040
            self.tile_size = 1024
            self.model_type = 'VGG19'
            # algorithmia giving weird errors when i dont use .h5 format
            self.deprecated_model_type = True

            self.classes = [
                '0-1',
                '11-20',
                '2-4',
                '21-40',
                '41-60',
                '5-10',
                '61-100',
                'Blank',
            ]

            # color maps. we supply the BGR values directly
            # light to dark red colors and white for blank
            # we do the following because colorpalette is sequential but our class points above is not so we have to reorder
            # the palette so that intensity of red matches percentage
            classes_no_blank = self.classes[:-1]

            # import matplotlib.pyplot as plt
            # cm = plt.get_cmap('Reds')
            # colors = [cm(i / len(classes_no_blank))[:-1] for i in range(len(classes_no_blank))]

            colors = sns.color_palette("coolwarm", len(classes_no_blank) + 1)
            colors.pop(3)  # removing the grey-ish color

            # colors = sns.color_palette("Reds", len(classes_no_blank))

            ideal = dict(zip(
                sorted(classes_no_blank, key=lambda x: int(x.split('-')[0])),
                colors
            ))
            self.colormaps = [ideal[c][::-1] for c in classes_no_blank] + [(1.0, 1.0, 1.0)]

            setup_config(self)


########################################################################################################################


def setup_config(config):
    '''
    Performs final setup for a config object

    :param config:
    :return:
    '''

    config.identity = config.type + '_' + config.folder_name
    config.io = io(config)

    if not hasattr(config, 'colormaps'):

        if hasattr(config, 'non_lesion_colormaps') and hasattr(config, 'lesion_color'):
            config.colormaps = [
                config.non_lesion_colormaps[c] if c in config.non_lesion_colormaps else config.lesion_color
                for c in config.classes]

        else:
            # assign default colormaps. classes contained in `whites` will have a white color
            color_gen = unique_colors.next_color_generator()
            whites = ['blank', 'marker']
            config.colormaps = ['white' if c.lower() in whites else next(color_gen) for c in config.classes]

    # indices setup. doesnt need to be in-order
    if hasattr(config, 'non_lesion_colormaps'):
        # make sure colormap non-lesion classes are present in classes
        if len(set(config.non_lesion_colormaps).difference(config.classes)):
            raise AlgorithmException(f'{set(config.non_lesion_colormaps).difference(config.classes)} not present in classes')
        config.non_lesion_indices = generate_class_indices(all=config.classes, sub=list(config.non_lesion_colormaps))
        config.non_lesion_classes = list(config.non_lesion_colormaps)


def generate_class_indices(all, sub):
    '''
    Generates which indices sub is within all
    ie if all=[a,b,c] and sub=[a,c] then the result is [0,2]

    :param c1: list
    :param c2: list
    :return:
    '''
    return np.nonzero(np.in1d(all, sub))[0]


def extract_model(model_uri):
    """
    Unzip model files from data collections
    """

    out_dir = '/tmp/unzipped_files'

    # Model path from data collections
    input_zip = settings.client.file(model_uri).getFile().name
    try:
        # Create directory to unzip model files into
        os.mkdir(out_dir)
        print("Created directory")
    except:
        print("Error in creating directory")
    zipped_file = zipfile.ZipFile(input_zip)
    # Extract unzipped files into directory created earlier
    zipped_file.extractall(out_dir)
    zipped_file.close()
    return out_dir
