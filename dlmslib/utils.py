import numpy as np


def is_integer(val):
    return isinstance(val, (int, np.int_))


def is_float(val):
    return isinstance(val, (float, np.float_))


def is_numeric(val):
    return is_float(val) or is_integer(val)


def is_list_of_float(val):
    if not isinstance(val, list):
        return False

    for ele in val:
        if not is_float(ele):
            return False

    return True


def is_list_of_string(val):
    if not isinstance(val, list):
        return False

    for ele in val:
        if not isinstance(ele, str):
            return False

    return True

