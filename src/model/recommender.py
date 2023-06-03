import sys
sys.path.append("..")

from src.model.KBRD import KBRD
from src.model.BARCOR import BARCOR
from src.model.UNICRS import UNICRS
from src.model.CHATGPT import CHATGPT

name2class = {
    'kbrd': KBRD,
    'barcor': BARCOR,
    'unicrs': UNICRS,
    'chatgpt': CHATGPT,
}

class RECOMMENDER():
    def __init__(self, crs_model, *args, **kwargs) -> None:
        model_class = name2class[crs_model]
        self.crs_model = model_class(*args, **kwargs)
        
    def get_rec(self, conv_dict):
        return self.crs_model.get_rec(conv_dict)
    
    def get_conv(self, conv_dict):
        return self.crs_model.get_conv(conv_dict)
    
    def get_choice(self, gen_inputs, option, state, conv_dict=None):
        return self.crs_model.get_choice(gen_inputs, option, state, conv_dict)