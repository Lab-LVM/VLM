import os
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageNet as TorchImagenet
from torchvision.transforms import transforms
from transformers import CLIPTokenizerFast

from . import VLMDataset, IMAGENET_CLASS_NAME


class ImageNet(VLMDataset, Dataset):
    dataset_path = 'imageNet'
    n_class = 1000

    def __init__(self, root, split='val', transform=None, target_transform=None, n_shot=0):
        dataset = TorchImagenet(os.path.join(root, self.dataset_path), split)
        class_name_list = IMAGENET_CLASS_NAME
        super().__init__(root, *self._imgs_targets(dataset), class_name_list, transform, target_transform, n_shot)

    @staticmethod
    def _imgs_targets(dataset):
        imgs = [x[0] for x in dataset.imgs]
        targets = dataset.targets
        return imgs, targets

    @property
    def prompt(self):
        return [
            lambda c: f'a bad photo of a {c}.',
            lambda c: f'a photo of many {c}.',
            lambda c: f'a sculpture of a {c}.',
            lambda c: f'a photo of the hard to see {c}.',
            lambda c: f'a low resolution photo of the {c}.',
            lambda c: f'a rendering of a {c}.',
            lambda c: f'graffiti of a {c}.',
            lambda c: f'a bad photo of the {c}.',
            lambda c: f'a cropped photo of the {c}.',
            lambda c: f'a tattoo of a {c}.',
            lambda c: f'the embroidered {c}.',
            lambda c: f'a photo of a hard to see {c}.',
            lambda c: f'a bright photo of a {c}.',
            lambda c: f'a photo of a clean {c}.',
            lambda c: f'a photo of a dirty {c}.',
            lambda c: f'a dark photo of the {c}.',
            lambda c: f'a drawing of a {c}.',
            lambda c: f'a photo of my {c}.',
            lambda c: f'the plastic {c}.',
            lambda c: f'a photo of the cool {c}.',
            lambda c: f'a close-up photo of a {c}.',
            lambda c: f'a black and white photo of the {c}.',
            lambda c: f'a painting of the {c}.',
            lambda c: f'a painting of a {c}.',
            lambda c: f'a pixelated photo of the {c}.',
            lambda c: f'a sculpture of the {c}.',
            lambda c: f'a bright photo of the {c}.',
            lambda c: f'a cropped photo of a {c}.',
            lambda c: f'a plastic {c}.',
            lambda c: f'a photo of the dirty {c}.',
            lambda c: f'a jpeg corrupted photo of a {c}.',
            lambda c: f'a blurry photo of the {c}.',
            lambda c: f'a photo of the {c}.',
            lambda c: f'a good photo of the {c}.',
            lambda c: f'a rendering of the {c}.',
            lambda c: f'a {c} in a video game.',
            lambda c: f'a photo of one {c}.',
            lambda c: f'a doodle of a {c}.',
            lambda c: f'a close-up photo of the {c}.',
            lambda c: f'a photo of a {c}.',
            lambda c: f'the origami {c}.',
            lambda c: f'the {c} in a video game.',
            lambda c: f'a sketch of a {c}.',
            lambda c: f'a doodle of the {c}.',
            lambda c: f'a origami {c}.',
            lambda c: f'a low resolution photo of a {c}.',
            lambda c: f'the toy {c}.',
            lambda c: f'a rendition of the {c}.',
            lambda c: f'a photo of the clean {c}.',
            lambda c: f'a photo of a large {c}.',
            lambda c: f'a rendition of a {c}.',
            lambda c: f'a photo of a nice {c}.',
            lambda c: f'a photo of a weird {c}.',
            lambda c: f'a blurry photo of a {c}.',
            lambda c: f'a cartoon {c}.',
            lambda c: f'art of a {c}.',
            lambda c: f'a sketch of the {c}.',
            lambda c: f'a embroidered {c}.',
            lambda c: f'a pixelated photo of a {c}.',
            lambda c: f'itap of the {c}.',
            lambda c: f'a jpeg corrupted photo of the {c}.',
            lambda c: f'a good photo of a {c}.',
            lambda c: f'a plushie {c}.',
            lambda c: f'a photo of the nice {c}.',
            lambda c: f'a photo of the small {c}.',
            lambda c: f'a photo of the weird {c}.',
            lambda c: f'the cartoon {c}.',
            lambda c: f'art of the {c}.',
            lambda c: f'a drawing of the {c}.',
            lambda c: f'a photo of the large {c}.',
            lambda c: f'a black and white photo of a {c}.',
            lambda c: f'the plushie {c}.',
            lambda c: f'a dark photo of a {c}.',
            lambda c: f'itap of a {c}.',
            lambda c: f'graffiti of the {c}.',
            lambda c: f'a toy {c}.',
            lambda c: f'itap of my {c}.',
            lambda c: f'a photo of a cool {c}.',
            lambda c: f'a photo of a small {c}.',
            lambda c: f'a tattoo of the {c}.',
        ]

    def _data_dict(self):
        train_dataset = TorchImagenet(os.path.join(self.root, self.dataset_path), 'train')
        train_data_dict = defaultdict(list)
        for i in range(len(train_dataset.imgs)):
            train_data_dict[train_dataset.targets[i]].append(train_dataset.imgs[i][0])

        return train_data_dict


RAND_AUG_TRANSFORMS = {
    'AutoContrast': 'auto contrasted',
    'Equalize': 'equalized',
    'Invert': 'inverted',
    'Rotate': 'rotated',
    'Posterize': 'posterized',
    'Solarize': 'solarized',
    'Color': 'colored',
    'Contrast': 'contrasted',
    'Brightness': 'brighter',
    'BrightnessIncreasing': 'more brighter',
    'Sharpness': 'sharper',
    'PosterizeIncreasing': 'more posterized',
    'SolarizeAdd': 'adding solarized',
    'SolarizeIncreasing': 'increasing solarized',
    'ColorIncreasing': 'color factor increased',
    'ContrastIncreasing': 'contrasted',
    'SharpnessIncreasing': 'more sharper',
    'ShearX': 'shear to x',
    'ShearY': 'shear to y',
    'TranslateXRel': 'translated by x',
    'TranslateYRel': 'translated by y',
}
AUG_PROMPT = [
    lambda augment, name: f'{augment} itap of a {name}.',
    lambda augment, name: f'itap of a {augment} {name}.',
    lambda augment, name: f'a bad {augment} photo of the {name}.',
    lambda augment, name: f'a {augment} origami {name}.',
    lambda augment, name: f'a {augment} {name} in a video game.',
    lambda augment, name: f'{augment} art of the {name}.',
    lambda augment, name: f'art of the {augment} {name}.',
    lambda augment, name: f'a {augment} photo of the {name}.',
    lambda augment, name: f'{augment} transformed image of {name}.',
    lambda augment, name: f'{augment} transformed photo of the {name}.',
]
RAND_AUG_TOKENS = {
    'AutoContrast': [5668, 9880, 1356],
    'Equalize': [2563, 6141],
    'Invert': [40709],
    'Rotate': [10055, 943],
    'Posterize': [22881, 7690],
    'Solarize': [1175, 40167],
    'Color': [11775],
    'Contrast': [9880, 1356],
    'Brightness': [18507],
    'BrightnessIncreasing': [750, 18507],
    'Sharpness': [1669, 1284],
    'PosterizeIncreasing': [750, 22881, 7690],
    'SolarizeAdd': [8676, 1175, 40167],
    'SolarizeIncreasing': [10270, 1175, 40167],
    'ColorIncreasing': [3140, 8124, 9156],
    'ContrastIncreasing': [9880, 1356],
    'SharpnessIncreasing': [750, 1669, 1284],
    'ShearX': [32108, 531, 343],
    'ShearY': [32108, 531, 344],
    'TranslateXRel': [20752, 638, 343],
    'TranslateYRel': [20752, 638, 344],
}
AUG_PROMPT_TOKENS = [
    lambda augment, name: [49406] + augment + [529, 2728, 539, 320] + name + [269, 49407],
    lambda augment, name: [49406, 529, 2728, 539, 320] + augment + name + [269, 49407],
    lambda augment, name: [49406, 320, 2103] + augment + [1125, 539, 518] + name + [269, 49407],
    lambda augment, name: [49406, 320] + augment + [31832] + name + [269, 49407],
    lambda augment, name: [49406, 320] + augment + name + [530, 320, 1455, 1063, 269, 49407],
    lambda augment, name: [49406] + augment + [794, 539, 518] + name + [269, 49407],
    lambda augment, name: [49406, 794, 539, 518] + augment + name + [269, 49407],
    lambda augment, name: [49406, 320] + augment + [1125, 539, 518] + name + [269, 49407],
    lambda augment, name: [49406] + augment + [17453, 2867, 539] + name + [269, 49407],
    lambda augment, name: [49406] + augment + [17453, 1125, 539, 518] + name + [269, 49407],
]
AND = [537]
IMAGENET_CLASS_NAME_TOKEN = [[1149, 634], [40293], [830, 1579, 7980], [6531, 7980], [15331, 1375, 7980],
                             [5031, 3077], [43229], [30926], [8047], [47640], [14815, 7097], [4999, 18523],
                             [1212, 18523], [1637, 1320], [21002, 30695], [2151, 7988], [2572, 10200], [4155],
                             [45973], [3170, 47054], [2151, 49317], [19867, 263, 3329, 539, 17131, 264],
                             [14875, 7517], [34593], [830, 5046, 9899], [1769, 19007, 3935], [8990, 37224],
                             [37224], [4533, 19007, 3935], [3165, 78, 5132, 331], [2151, 4537, 11438],
                             [2677, 11438], [20626, 11438], [549, 12367, 1375, 2102, 10912],
                             [21417, 893, 2102, 10912], [11673, 10912], [22366, 5164], [2063, 10912],
                             [37804, 38616], [1901, 21279, 1388], [7027, 514, 1947],
                             [7301, 11584, 973, 573, 1719, 4132, 17221], [602, 1696],
                             [936, 2527, 268, 557, 3441, 17221], [28574, 17221], [46952, 6060],
                             [4759, 1901, 17221], [41317], [6686, 40993, 5471], [24252, 24757], [2151, 28574],
                             [9511, 517, 527, 3054], [10945, 8798], [2540, 268, 557, 3441, 8798],
                             [6311, 13255, 268, 578, 1252, 8798], [8990, 1901, 8798], [2365, 8798],
                             [48420, 8798], [1573, 8798], [7721, 8798], [930, 8798], [14732, 616, 2607, 1208],
                             [4736, 2172, 13370], [3606, 21574], [1901, 44189], [2102, 8798],
                             [44255, 34526, 27401], [6311, 18535, 893, 40824, 8798], [5869, 46245, 40824, 8798],
                             [40626, 1411, 802], [44495, 786], [28461], [4481, 2756, 7622], [10942, 7622],
                             [4759, 2756, 7622], [5036, 1449, 19949], [32370, 36332], [5916, 7622], [15617],
                             [48889, 19560], [1449, 36327], [79, 2002, 714, 1670], [681, 8448, 36327],
                             [14753, 36327], [661, 702, 9899], [34392], [34872], [4736, 5046, 23565],
                             [1961, 5557], [47659, 268, 36656, 622, 916, 527, 2822], [20164, 643, 875],
                             [1368, 6372], [5028, 24060], [15771, 2886], [33072], [2293, 19607], [1879, 753],
                             [6910], [736, 268, 7318, 1356, 1316, 1822, 2771], [13822], [1449, 12530],
                             [764, 33600], [68, 1337, 8790], [3172, 88, 13841], [26007, 638], [36654],
                             [1087, 6575], [30988], [2102, 49019], [4812, 12054], [8986, 10945],
                             [557, 13127, 654], [616, 634], [23132], [34190], [2102, 34190], [13515, 525],
                             [1290, 9193, 5955, 6124, 718], [2616, 5822, 632, 11574], [2172, 11574],
                             [25218, 1803, 11574], [736, 674, 11574], [2151, 13793], [6288, 344, 13793],
                             [24184, 2759], [43953, 11574], [2462, 7446], [1579, 46452], [1449, 46452],
                             [27001, 2886], [30323], [1274, 1746, 22593], [830, 42002], [3010, 12959, 3329],
                             [14626, 3329], [554, 1037, 2667], [4176, 3011, 21131, 579], [2151, 1664, 339],
                             [31299, 746], [681, 7876, 5522, 2441], [2616, 2424], [4176, 1893, 24685],
                             [639, 33684], [40545, 15965], [34131], [674, 14952], [42797, 2158], [5046, 11650],
                             [6613, 11650], [22743, 846], [2102, 5567], [30181], [4925, 8979], [39838],
                             [661, 1442, 619, 611], [823, 327, 34354], [674, 5127, 35435], [1884, 20516],
                             [5988, 14455], [39302, 6221, 16580, 893], [15919, 13561], [1244, 1167, 13561],
                             [30044], [48480, 853], [3829, 15617, 34260, 13561],
                             [1449, 537, 5039, 34260, 13561], [975, 7702, 6150, 34260, 13561],
                             [3469, 5007, 13561], [1893, 6195, 34260, 13561], [1141, 4397, 328],
                             [4889, 9453, 13561], [5175, 27363], [4295, 20744], [72, 8212, 550, 13561],
                             [15971, 544, 2166, 853], [34710, 13561], [980, 9009], [7442, 19454, 13561],
                             [6958, 44732, 528], [29198, 6029, 14455], [2151, 29198, 14455],
                             [4152, 9002, 14455], [6177, 14455], [13097, 1746, 14455], [4889, 14455],
                             [12202, 14455], [15812, 14455], [8633, 14455], [7794, 3240, 14455], [34306, 14455],
                             [567, 6468, 1714, 14455], [2014, 1893, 1336, 14455], [42467, 14455], [6258, 14455],
                             [2257, 2082, 1462, 5382, 14455], [4399, 14455], [16377, 45745, 3716],
                             [4687, 45745, 3716], [5807, 45745, 3716], [7442, 14455], [24057, 14455],
                             [6258, 29554, 14455], [3773, 268, 23401, 20505, 576, 14455],
                             [1593, 16062, 1579, 14455], [75, 560, 1344, 688, 706], [6313, 268, 23401, 28394],
                             [20795, 268, 23401, 28394], [3878, 28394], [25409, 28394], [32909, 2174, 28394],
                             [4710, 37397, 21938, 29883], [28251, 26707], [3469, 23153], [4889, 23153],
                             [8244, 23153], [18818, 1929], [996, 1034, 35435], [3469, 30117, 35435],
                             [10977, 30117, 35435], [622, 4744, 35435], [13266, 35435], [4889, 1573, 35435],
                             [1836, 5669, 345], [13731, 6719, 618], [1464, 524, 524, 925, 832, 1929],
                             [1662, 4211, 533], [1036, 746], [6258, 2825, 5319], [6686, 521, 8695],
                             [896, 3469, 23604, 1929], [29595, 23604, 1929], [30611], [6177, 30611],
                             [3728, 11987, 1437, 1082, 25197, 1929], [34806, 42203], [4710, 13242, 1929],
                             [639, 867, 4783], [16377, 2629, 21299], [7033, 9287, 3965, 1929],
                             [867, 32220, 3965, 1929], [882, 25805, 4465, 1057, 17817, 1616, 323],
                             [724, 47030, 3466, 1057, 17817, 1616, 323], [15363], [4537, 42311], [24057, 42311],
                             [3461, 15611], [830, 19330], [545, 269, 14579], [20421], [33898, 1662, 548, 3130],
                             [34058, 20421], [44275, 550], [702, 5509, 2629, 21299], [1244, 524, 2697], [17739],
                             [6581, 13873], [28079, 1929], [830, 39744, 1929], [30006, 3301], [35156, 9945],
                             [21657, 21657], [9724, 552, 2636], [11020, 17855, 525], [40044, 10977, 31320],
                             [30501, 10977, 31320], [5988, 34961], [16377, 34961], [5807, 34961],
                             [8186, 4658, 1285, 1929, 263, 87, 14628, 14544, 729, 1828, 4250, 264],
                             [5046, 5916], [33898, 39899, 5916], [736, 5916, 541, 723, 538, 5916], [26586],
                             [7545, 334], [67, 5341], [4736, 3220, 1929], [1441, 3452], [736, 3240],
                             [4274, 3240], [9138, 3240], [5046, 3240], [36145, 2368], [6531, 2368],
                             [19859, 2368], [43161, 2368], [12428, 23033], [27042], [28941], [15931],
                             [2583, 15931], [14757], [5567], [6531], [27431], [2866, 4298], [2151, 1449, 4298],
                             [12214, 4298], [30007, 4298], [749, 13822], [26714, 9341], [6531, 16534], [40038],
                             [2461, 16534], [41047, 16534], [7232, 16534], [33712, 16534],
                             [41744, 1364, 1299, 16534], [716, 3000], [3228], [5028], [773], [48894],
                             [5373, 21297], [4987, 21297], [622, 916, 31073], [11550, 39946], [14539, 1886],
                             [9377, 21748], [30012, 1340], [32824], [1880, 1000, 3228], [736, 21013, 9738],
                             [8318, 1094, 9738], [22619, 9738], [2442, 1579, 9738], [47659, 9738],
                             [39734, 16147, 268, 24993, 9738], [44283], [2102, 565, 8979], [2102, 19787],
                             [22218, 4132, 10274], [18464], [883, 4301, 10274], [33313], [817, 5059, 715],
                             [3240, 14004], [675, 15452], [22874], [18537, 9619], [4176, 1852, 5233, 4558],
                             [22548], [9619], [3220, 35473], [984, 1325, 326], [28398, 45825, 718], [12071],
                             [1573, 8054], [19741], [2007, 263, 7115, 2801, 9629, 264], [18106, 8130, 9629],
                             [17055, 72, 24748], [915, 600, 571, 1509], [36641, 263, 39177, 264], [4837, 4765],
                             [24663, 21914], [36679], [44308], [42017], [4759, 628, 944, 536],
                             [1449, 268, 38040, 1765, 10048], [22456], [42194], [22363], [37897, 7588],
                             [2097, 268, 580, 538, 30007], [36112, 5039], [21994], [10543, 1072, 14080],
                             [18964, 525], [29686, 2704], [700, 524, 525], [1300, 601, 9465], [1202, 7100],
                             [1961, 37268], [4095, 337], [1449, 268, 537, 268, 1579, 7300, 2840],
                             [644, 647, 663, 533, 9465], [675, 617, 1167], [1579, 268, 6153, 44241, 8979],
                             [7470, 1803, 9465], [48504, 9465], [20629, 626, 344, 568, 7622, 9465],
                             [4176, 14004, 9465], [2540, 268, 20626, 534, 18293], [512, 28833], [7128, 10299],
                             [4736, 6867, 10299], [736, 12952], [4687, 12952], [8567, 2092, 2759], [27462],
                             [3467, 9137], [2172, 2488, 2759], [18386, 77, 2759], [31580], [5824, 2759],
                             [8872, 2759], [755, 27369, 2759], [596, 33710], [596, 5917], [7935, 13719],
                             [48760], [10616, 5084], [7706, 12308], [1281, 11618], [1281, 1158], [16385],
                             [15555], [45206, 5299], [19963, 6716], [23342, 856], [29502], [9798, 753],
                             [9679, 15354], [14894], [13377], [6827, 13663], [13634], [3807, 2301, 5356],
                             [1963, 268, 5182], [27014], [38093, 881, 270, 1722, 6879], [1040, 3718],
                             [12296, 4269], [37707], [10942], [1040, 20060], [11703], [7360, 24983], [3470],
                             [3835], [22831, 8121], [1244, 1745], [7411, 3938], [5713, 15953], [39942],
                             [2631, 13260], [13717], [571, 3074],
                             [4323, 3801, 263, 6375, 3575, 541, 8385, 334, 264], [2544, 5392], [2544, 3313],
                             [3718, 4730], [1794, 21022], [33756, 11652], [14531], [2540, 30541],
                             [29172, 45719], [6908, 1212], [45461, 7603], [8444, 30865], [40236, 3422],
                             [23186, 30636], [2828, 2068], [17999], [5392, 3938], [7507, 4040], [4040, 3422],
                             [11655, 4642, 16097], [5860], [2206, 1573], [12930, 5135], [28008], [10498],
                             [21948], [43212, 12473], [1400, 268, 4163, 3231], [18732, 1557], [4581, 535, 3937],
                             [23460, 23360], [12674], [15661], [23503], [753, 8998], [30501], [1615, 7220],
                             [36665], [5999, 4274], [19845, 2063, 270, 41812], [1615, 6744],
                             [19022, 20415, 4169], [19717], [19717, 2477], [3540], [1481, 6424, 550],
                             [4480, 2477], [23013], [3451, 1951], [3946], [3946, 268, 2468, 12679],
                             [3946, 2614], [37617], [6824, 10563], [27407, 4385, 528],
                             [3718, 541, 3630, 1337, 614], [2817, 9937], [1677, 17335], [2735], [2016, 6128],
                             [1321, 11403], [10625, 37098], [36528], [1194, 1345], [622, 916, 4132, 26210],
                             [2453, 9722], [12898, 4836], [19873, 541, 25307], [11312, 4381],
                             [11639, 652, 13017], [6417, 2183], [14913, 1158], [19608], [18148, 14225],
                             [851, 2315], [13657, 8087], [13657, 3801], [29384], [4585, 14626], [5033, 11122],
                             [24756], [19192, 2722], [1192, 868, 5168], [1192, 8513, 1069], [1815, 4048],
                             [729, 4923, 632], [4926], [6550], [14345, 11639, 652], [14654, 11381, 17243],
                             [34382], [2794, 6716], [2794, 1239], [8658, 2175], [11039, 14559], [40524],
                             [10712, 14937], [8997], [1929, 28438], [9827], [7188, 9063], [18634, 13871],
                             [8698], [13353, 4987], [15901, 3718], [7991, 12579], [5031, 2261], [5031, 5084],
                             [5031, 30439], [6166, 2119], [27358], [17098, 4169], [1710, 9674], [17871, 14732],
                             [16576, 9937], [2951, 4440], [1769, 4629], [1769, 3750], [11536, 8170], [23746],
                             [15464, 4269], [1882, 11122], [43202], [13405], [13405, 5356],
                             [2721, 268, 3574, 2722], [17023, 1615], [3461, 9607], [38516, 7437], [9686, 7356],
                             [13760, 4629], [2474, 8306, 541, 20771, 1962], [2474, 9173], [29559, 1094],
                             [861, 268, 21310], [3096, 1069], [3096, 13065], [1854, 38005], [28486], [13719],
                             [2991, 7894], [20819], [39372, 34748], [13826, 2183], [5008, 5132, 715],
                             [2225, 9289], [2225, 10449], [2349, 268, 2700], [9401], [27692], [2225, 24425],
                             [2463, 268, 4042, 11639, 652], [1722, 2352, 3455], [1626, 17970, 2620],
                             [10828, 2576], [27339], [12919, 6405, 881], [3447, 9663], [1187, 881],
                             [1137, 6128], [48583], [10395], [19573, 12386], [13517, 1561, 25775, 2411],
                             [4558, 268, 8893, 5299], [14930, 3313], [17889], [7080, 5391], [15127, 8842],
                             [10157], [11286], [339, 268, 2523], [32987, 12875], [6237, 6805], [6308, 4987],
                             [40024], [9897, 7601], [18412], [4352, 7356], [572, 3287], [26700, 10097],
                             [10464, 11639, 652], [11024, 30895], [8666, 3938], [4861, 8998], [3519], [37404],
                             [14102], [47287, 715], [4918, 11618], [15618], [12105, 268, 525, 7342], [25260],
                             [1179, 4914], [2207, 956, 24297, 5294, 3313], [12429, 6637], [13838, 19438],
                             [17389, 3365], [31482], [29581], [637, 268, 2754, 20144, 3940], [723, 5341, 2202],
                             [675, 9213], [675, 732, 1792], [8306], [7733, 4987], [1686, 8170], [21988],
                             [18064, 1937], [5616, 9937], [37650, 1757], [24643], [24240, 12579], [4323, 11075],
                             [5205, 753], [1810, 2840], [33409, 3476, 339], [1810, 2451], [14411], [76, 4343],
                             [15588, 3814], [3451, 1137], [1623, 2863, 339], [617, 7825], [21850], [10198],
                             [617, 2966], [27258, 537, 661, 7698], [8060, 3938], [12694], [25083, 2315],
                             [46029], [3965, 3701], [10782], [11639, 652, 9301], [33032, 9677], [3584, 2451],
                             [670, 7873], [4044, 9059], [6906, 9183], [7385], [1794, 5699, 10598],
                             [16365, 11639, 652], [78, 1308, 30171], [78, 21773], [78, 44357], [78, 8515, 652],
                             [2870, 7541], [10459, 13213], [5092, 34153, 7979], [1658, 12386], [33198, 13065],
                             [16214, 8306], [4306, 25022, 270, 11658], [20811], [20811, 6744], [3798, 4381],
                             [7948, 8594], [37231], [6381], [7437, 23746], [2802, 15953], [30122],
                             [18201, 8616], [1452, 8771], [5984, 10104], [17080, 1615], [14304], [2642, 1951],
                             [12862, 5535], [11200, 2068], [11200, 41465, 528], [17525], [661, 9618, 4531],
                             [1153, 622, 11348], [926, 66, 11399], [901, 2825, 5152, 655], [33559, 12679],
                             [15382, 4629], [11348], [28245, 2723], [19226, 5392], [12182],
                             [2089, 268, 17710, 1069], [2629, 6744], [13592, 1158], [3979, 13445], [4593, 5363],
                             [16833, 11809], [6102, 3365], [5135, 12034], [3985, 38363], [15122, 1502],
                             [30427, 3934], [8170], [2120, 2451], [3809, 5150], [2831, 2175], [13533, 5392],
                             [3912, 5168], [9026, 568, 6744], [1807, 11626], [6583, 19220], [14521], [6622],
                             [14411], [27816], [4111, 17550], [33468, 3365], [16480], [48951], [23977],
                             [2471, 1615], [37691], [39372], [2638], [2638, 19037], [2443, 11703],
                             [24329, 5299], [4935, 7087, 17166], [36050, 3934], [36662], [9687, 3366], [4489],
                             [38747], [15354], [7081, 4269], [532, 28155, 7005], [517, 7209], [4964, 1069],
                             [26286, 18064, 4987], [24781], [2996], [3406, 5164], [6611, 26210], [42185],
                             [1808, 846], [32296], [31716, 17514], [24455, 5879], [1228, 2840], [2493, 6071],
                             [30104], [43877, 10198], [14225], [48748], [4922, 7373], [15974, 4169], [8670],
                             [7342, 2183], [719, 2697, 3750, 270, 1530, 8375, 801], [4266, 10338],
                             [4266, 13065], [31778], [7777, 3938], [7777, 17223], [3428],
                             [1398, 546, 16765, 3428, 8306], [6982, 3365], [8837, 6175], [21468, 2489],
                             [14500, 4169], [16590, 7553], [3880, 3451], [3880, 38363], [11088, 40116],
                             [4233, 1069], [18277], [5199, 13689, 13522], [734, 47781], [7077, 3814],
                             [13017, 2138, 2411], [2138, 23246], [2138, 15842], [588, 1744, 320], [3975, 4440],
                             [7622, 4601], [6288, 3287], [2054, 1615], [7901], [2170], [6972, 30439],
                             [1417, 8557, 2465], [4726, 8698], [795, 1325, 7979], [13365], [2441, 2569],
                             [6005, 1239], [21423], [1894, 5155], [17500], [12265, 3466], [12724], [858, 2217],
                             [19968], [3940], [968, 11381], [12906], [12906], [29355], [15417, 2465], [33900],
                             [22442], [7681, 38531, 270, 9680], [7429], [12176, 5893], [974, 81, 7071],
                             [2175, 10725], [6172], [5749, 2477], [40749], [11798, 4298], [8608], [5298, 1069],
                             [6303, 2146, 6449], [2184, 17223], [42331, 1059], [1607, 1743, 4169], [15999],
                             [8227, 6449], [39061], [13905, 1557], [11071, 4922], [15820], [45376, 8170],
                             [18653, 4629], [5988, 2183], [14607], [6684, 268, 4700, 4629], [14621],
                             [23846, 7356], [9511, 38089], [924, 675, 550], [36141], [14782, 1037, 8708, 8557],
                             [10306, 2565, 2840], [44423], [2069, 15450], [5522, 522, 989], [42180, 13017],
                             [17143], [7648, 38089], [29818, 7894], [18420, 15619], [20431],
                             [27469, 775, 541, 516, 2146, 12374], [11063, 10033], [29202, 4169], [31657, 777],
                             [39113], [17058], [7458], [20375, 5391], [2569, 6716], [12154], [15020],
                             [4323, 7706], [14551], [13029, 4169], [1573, 5392], [1573, 27619], [1573, 4730],
                             [11610, 27619], [19746], [2225, 12216], [4879, 3750], [4879, 10097], [12884, 3422],
                             [2604, 5392], [16451, 1340], [28912], [9057, 14024], [13283],
                             [9109, 268, 6879, 12679], [48878], [7528, 4440], [88, 24309], [3364], [4962, 1116],
                             [32945], [3399, 541, 2012, 2292], [3399, 1395], [8349, 6164], [6225], [5135],
                             [40115], [616, 30625], [2069, 5168], [924, 44964], [733, 3867], [38481, 579],
                             [45334], [28777], [36707], [41200], [2069, 1929], [27395, 11567], [18407], [19094],
                             [23802], [29873], [18440, 13312], [37537, 13312], [36867, 13312], [19787], [39647],
                             [3718, 8253], [811, 36612], [13011], [22710, 2915, 3055], [10233], [4287], [7184],
                             [20719], [14831], [8922], [2679, 5190], [26471, 617, 1430, 263, 26516, 3055, 264],
                             [32415], [11367], [14038, 3340], [3820, 16611], [14983], [10805, 18273], [4474],
                             [5168, 5319], [23473], [736, 2604], [17098], [3274, 1937], [13778, 31157], [3965],
                             [10799], [10625], [12054, 15624], [619, 33121], [42595], [644, 749, 4214],
                             [2147, 2411], [2117], [3136], [14581], [3470, 2477], [18342, 22813],
                             [20559, 22094], [46099, 6488], [12865], [4481, 2909, 568, 39644], [5894], [37537],
                             [3568, 6584], [4558, 21834, 6488], [12054, 30677], [13199, 1029],
                             [2168, 532, 47703], [2588, 74, 9607, 13011], [3475, 1565, 30677],
                             [8047, 539, 518, 6267, 13011], [647, 34078], [5894, 32863], [11071, 2802]]
MAX_LENGTH = 77


class RandAugment:
    def __init__(self, ops, num_layers=2, choice_weights=None):
        self.ops = ops
        self.num_layers = num_layers
        self.choice_weights = choice_weights
        self.replace = self.choice_weights is None

    def __call__(self, img):
        # no replacement when using weighted choice
        ops = np.random.choice(
            self.ops,
            self.num_layers,
            replace=self.replace,
            p=self.choice_weights,
        )
        for op in ops:
            img = op(img)
        return img, ops

    def __repr__(self):
        fs = self.__class__.__name__ + f'(n={self.num_layers}, ops='
        for op in self.ops:
            fs += f'\n\t{op}'
        fs += ')'
        return fs


class ImageNetRandaugPrompt(ImageNet):
    def __init__(self, root, split='val', transform=None, target_transform=None, n_shot=0, num_classes=1000):
        super().__init__(root, split, transform, target_transform, n_shot)
        self.len_prompt = len(AUG_PROMPT_TOKENS)
        self.num_classes = num_classes

    def setup_prompt_transform(self):
        self.pre_processing = transforms.Compose(self.transform.transforms[:2])
        self.randaug = RandAugment(**self.transform.transforms[2].__dict__)
        self.post_processing = transforms.Compose(self.transform.transforms[3:])

    def ra_prompt(self, idx, ra_tf, target):
        prompt = AUG_PROMPT_TOKENS[idx % self.len_prompt]
        rand_augmentation_token = RAND_AUG_TOKENS[ra_tf[0].name] + AND + RAND_AUG_TOKENS[ra_tf[1].name]

        prompt = prompt(rand_augmentation_token, IMAGENET_CLASS_NAME_TOKEN[target])
        return prompt + [49407] * (MAX_LENGTH - len(prompt))

    def __getitem__(self, idx):
        path, target = self.imgs[idx], self.targets[idx]
        imgs = self.loader(path)

        imgs = self.pre_processing(imgs)
        imgs, ra_tf = self.randaug(imgs)
        imgs = self.post_processing(imgs)

        return imgs, target, torch.tensor(self.ra_prompt(idx, ra_tf, target), dtype=torch.int64)


class ImageNetRandaugPromptV2(ImageNet):
    def __init__(self, root, split='val', transform=None, target_transform=None, n_shot=0):
        super().__init__(root, split, transform, target_transform, n_shot)
        self.len_prompt = len(AUG_PROMPT_TOKENS)

    def setup_prompt_transform(self):
        self.pre_processing = transforms.Compose(self.transform.transforms[:2])
        self.randaug = RandAugment(**self.transform.transforms[2].__dict__)
        self.post_processing = transforms.Compose(self.transform.transforms[3:])

    def ra_prompt(self, idx, ra_tf, target):
        prompt = AUG_PROMPT_TOKENS[idx % self.len_prompt]
        rand_augmentation_token = RAND_AUG_TOKENS[ra_tf[0].name] + AND + RAND_AUG_TOKENS[ra_tf[1].name]

        prompt = prompt(rand_augmentation_token, IMAGENET_CLASS_NAME_TOKEN[target])
        return prompt + [49407] * (MAX_LENGTH - len(prompt))

    def original_prompt(self, idx, target):
        prompt = AUG_PROMPT_TOKENS[idx % self.len_prompt]
        prompt = prompt([3117], IMAGENET_CLASS_NAME_TOKEN[target])
        return prompt + [49407] * (MAX_LENGTH - len(prompt))

    def __getitem__(self, idx):
        path, target = self.imgs[idx], self.targets[idx]
        img = self.loader(path)

        img = self.pre_processing(img)
        ra_img, ra_tf = self.randaug(img)
        ra_img = self.post_processing(ra_img)
        img = self.post_processing(img)

        return img, ra_img, target, torch.tensor(self.original_prompt(idx, target), dtype=torch.int64), torch.tensor(self.ra_prompt(idx, ra_tf, target), dtype=torch.int64)



if __name__ == '__main__':
    ds = ImageNetRandaugPrompt('/data', transform=transforms.ToTensor(), n_shot=0)
    tokenizer = CLIPTokenizerFast.from_pretrained('openai/clip-vit-base-patch32')

    augment_prompt = [
        lambda augment, name: f'{augment} itap of a {name}.',
        lambda augment, name: f'itap of a {augment} {name}.',
        lambda augment, name: f'a bad {augment} photo of the {name}.',
        lambda augment, name: f'a {augment} origami {name}.',
        lambda augment, name: f'a {augment} {name} in a video game.',
        lambda augment, name: f'{augment} art of the {name}.',
        lambda augment, name: f'art of the {augment} {name}.',
        lambda augment, name: f'a {augment} photo of the {name}.',
        lambda augment, name: f'{augment} transformed image of {name}.',
        lambda augment, name: f'{augment} transformed photo of the {name}.',
    ]

    size = len(RAND_AUG_TRANSFORMS) - 1
    aug_size = len(augment_prompt) - 1
    import random
    from tqdm import tqdm

    for _ in tqdm(range(1000)):
        i1, i2 = random.randint(0, size), random.randint(0, size)
        p1 = random.randint(0, aug_size)

        aug1 = list(RAND_AUG_TRANSFORMS.keys())[i1]
        aug2 = list(RAND_AUG_TRANSFORMS.keys())[i2]
        prompt = augment_prompt[p1]

        ra_fs = ''
        ra_fs += f'{RAND_AUG_TRANSFORMS[aug1]} and '
        ra_fs += f'{RAND_AUG_TRANSFORMS[aug2]}'
        prompt = prompt(ra_fs, 'X')

        t_rafs = RAND_AUG_TOKENS[aug1] + AND + RAND_AUG_TOKENS[aug2]
        t_prompt = AUG_PROMPT_TOKENS[p1](t_rafs, [343])
        t_prompt = t_prompt + [49407] * (MAX_LENGTH - len(t_prompt))

        tokenized = list(tokenizer(prompt, return_tensors='np', padding='max_length')['input_ids'][0])

        if not t_prompt == tokenized:
            print(prompt)
            print(t_prompt)
            print(tokenized)
            break
