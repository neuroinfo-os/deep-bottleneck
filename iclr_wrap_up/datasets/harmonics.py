"""The dataset mostly used in the original Tishby paper."""
from iclr_wrap_up import utils

def load():
    train, test = utils.get_IB_data('2017_12_21_16_51_3_275766')
    # Instantiate iterable.
    return train, test

