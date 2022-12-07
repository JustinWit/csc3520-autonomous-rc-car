# read_data.py
'''method for loading data from donkey tubs'''
import os
import pdb
import json
import numpy as np
import cv2
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_data(path):
    dpath = os.path.join(ROOT, path)
    mpath = os.path.join(dpath, 'manifest.json')
    f = open(mpath, 'r')
    line1 = f.readline()
    line2 = f.readline()
    f.readline()
    line4 = f.readline()
    x = f.read()
    f.close()
    data = json.loads(x)

    # data['paths'] gives catalog names
    # data['deleted_indexes'] for removed records
    # data['current_index] for size of dataset

    n = data['current_index']
    max_len = data['max_len']

    # np array to store data
    img_arr = np.zeros((n, 120, 160, 3))
    out_arr = np.zeros((n, 2), dtype=float)

    # loop over catalogs to extract info for data
    for i in data['paths']:
        cpath = os.path.join(dpath, i)
        f = open(cpath, 'r')
        
        # loop over all entrys in catalog
        for j in range(max_len):
            l = f.readline()

            # end of file
            if not l:
                break

            # read entry
            entry = json.loads(l)
            img_path = os.path.join(dpath,'images', entry['cam/image_array'])
            index = entry['_index']

            # get data for index
            img = cv2.imread(img_path)
            throttle = entry['user/throttle']
            angle = entry['user/angle']

            # add data to arrays
            img_arr[index] = img
            out_arr[index, 0] = throttle
            out_arr[index, 1] = angle
        # print updates
        print(f'{i} finished')

    pdb.set_trace()
    return img_arr, out_arr



if __name__ == '__main__':
    load_data('rc-car\\data')