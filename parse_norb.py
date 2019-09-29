# Dataset was saved in the same dir with this script
import numpy as np
import os
from PIL import Image


def getPath(which_set, filetype, dirname='data'):
    """
    Getting the path for the desired dataset.
    which_set: train, test
    filetype: dat, cat, info
    """     
    if which_set == 'train':
        instance_list = '46789'
    elif which_set == 'test':
        instance_list = '01235'
    filename = 'smallnorb-5x%sx9x18x6x2x96x96-%s-%s.mat' % \
        (instance_list, which_set + 'ing', filetype)
    return os.path.join(dirname, filename)


def readNums(file_handle, num_type, count):
    """
    Reads 4 bytes from file, returns it as a 32-bit integer.
    """
    num_bytes = count * np.dtype(num_type).itemsize
    string = file_handle.read(num_bytes)
    return np.fromstring(string, dtype=num_type)


def readHeader(file_handle, debug=False):
    """
    Reads the header of the file.
    file_handle: an open file handle.
    returns: data type, element size, rank, shape, size
    """

    key_to_type = {0x1E3D4C51: ('float32', 4),
                   # 0x1E3D4C52 : ('packed matrix', 0),
                   0x1E3D4C53: ('float64', 8),
                   0x1E3D4C54: ('int32', 4),
                   0x1E3D4C55: ('uint8', 1),
                   0x1E3D4C56: ('int16', 2)}

    type_key = readNums(file_handle, 'int32', 1)[0]
    elem_type, elem_size = key_to_type[type_key]
    if debug:
        print("header's type key, type, type size: ",
              type_key, elem_type, elem_size)
    if elem_type == 'packed matrix':
        raise NotImplementedError('packed matrix not supported')

    num_dims = readNums(file_handle, 'int32', 1)[0]
    if debug:
        print('# of dimensions, according to header: ', num_dims)

    shape = np.fromfile(file_handle,
                        dtype='int32',
                        count=max(num_dims, 3))[:num_dims]

    if debug:
        print('Tensor shape, as listed in header:', shape)

    return elem_type, elem_size, shape


def parseNORBFile(file_handle, debug=False):
    """
    Parse file into numpy array and return.
    file_handle: an open file handle.
    """
    elem_type, elem_size, shape = readHeader(file_handle, debug)
    beginning = file_handle.tell()
    num_elems = np.prod(shape)
    result = np.fromfile(file_handle,
                         dtype=elem_type,
                         count=num_elems).reshape(shape)
    return result


def get_data(which_set='train', file_type='dat', path='data', debug=True):
    file_path = getPath(which_set, file_type, path)
    file_handle = open(file_path, 'rb')
    norb_data = parseNORBFile(file_handle, debug)
    return norb_data
