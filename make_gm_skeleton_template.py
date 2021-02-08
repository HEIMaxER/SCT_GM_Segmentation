import argparse
import time
import pickle

import numpy as np

from SCT_segmentation import data as dm
from SCT_segmentation import GM_analysis as gma

from skimage.morphology import skeletonize

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-gt", "--gt_folder", help="Input folder with all scans and ground truth masks")
    parser.add_argument("-o", "--output", help="Output folder for the SCT GM mask scores")
    args = parser.parse_args()

    sct_tree = dm.load_files(args.gt_folder)

    CROP_SIZE = (50,50)

    avg_masks = {
        'site1': np.zeros(CROP_SIZE),
        'site2': np.zeros(CROP_SIZE),
        'site3': np.zeros(CROP_SIZE),
        'site4': np.zeros(CROP_SIZE),
        'average': np.zeros(CROP_SIZE),
    }

    start_time = time.time()

    for site in sct_tree.keys():
        print("Getting skeleton for "+site+" images")
        sc_nbr = len(sct_tree[site].keys())
        k=1
        for sc in sct_tree[site].keys():
            print("Getting skeletons of scan "+str(k)+" out of "+str(sc_nbr))

            for sl in range(sct_tree[site][sc]['slice_nbr']):

                cropped_mask = gma.get_cropped_gm_mask(sct_tree[site][sc], sl, 2, CROP_SIZE)
                if len(cropped_mask) > 1:
                    avg_masks[site] = np.where(cropped_mask==1., 1, avg_masks[site])
            k+=1
        avg_masks['average'] = np.where(avg_masks[site]==1., 1, avg_masks['average'])
        avg_masks[site] = skeletonize(avg_masks[site])

    avg_masks['average'] = skeletonize(avg_masks['average'])
    end_time = time.time()
    dur = end_time-start_time

    output_file = open(args.output+"average_masks.pkl", "wb")
    pickle.dump(avg_masks, output_file)
    output_file.close()

    print("All done !")
    print("Total duration : "+str(int(dur//60))+"min and "+str(int(dur%60))+"s")

    print("Saved tempaltes at : "+args.output+"average_masks.pkl")
if __name__ == "__main__":
    main()