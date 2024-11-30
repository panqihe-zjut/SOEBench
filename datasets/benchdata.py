import sys
from torch.utils.data import Dataset
sys.path.append('./')

from dataset.coco       import coco
from dataset.openimage  import openimage

valcoco       =  './annotations/annotations-filter_coco2017_validation.json'
valopenimage  =  './annotations/annotations-filter_openimage_validation.json'
testopenimage =  './annotations/annotations-filter_openimage_test.json'

class benchdata(Dataset):
    def __init__(self,  dataname, transformFlag, thred_size, labeltype):
        if dataname=='openimageval':
            self.dataset = openimage(annotations_file=valopenimage, transformFlag=transformFlag, thred_size=thred_size, labeltype=labeltype)
        elif dataname=='openimagetest':
            self.dataset = openimage(annotations_file=testopenimage, transformFlag=transformFlag, thred_size=thred_size, labeltype=labeltype)
        elif dataname=='cocoval':
            self.dataset =      coco(annotations_file=valcoco, transformFlag=transformFlag, thred_size=thred_size, labeltype=labeltype)