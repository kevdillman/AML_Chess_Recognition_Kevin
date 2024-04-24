import os

from skimage.io import imread
from skimage.util import view_as_blocks
import numpy as np
import sys

def runt(fileName, src, fen_from_filename, fen_from_position, NUM_SPACES_PER_BOARD):
    print('Hello from the child process')
    # flush output
    sys.stdout.flush()

    dimension = 400 // 8
    data = {'fenstring': [], 'data': []}
    spaceTilesStored, count = 0, 0

    if fileName.endswith('.jpg') or fileName.endswith('.jpeg'):
        im = imread(os.path.join(src, fileName))
        fenString = fen_from_filename(fileName)

        patches = view_as_blocks(im, block_shape=(dimension, dimension, 3)).reshape(-1, dimension, dimension, 3)
        boardPosition = 0
        
        for patch in patches:
            patchType = fen_from_position(boardPosition, fenString)
            if patchType == ' ':
                if spaceTilesStored >= NUM_SPACES_PER_BOARD:
                    continue
                spaceTilesStored += 1
            
            #hog_features = HogTransform(patch)
            data['data'].append(patch)
            data['fenstring'].append(patchType)
            boardPosition += 1
    
    return data
