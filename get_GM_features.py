import argparse
import time

import pickle as pkl
import numpy as np
import pandas as pd

from SCT_segmentation import data as dm
from SCT_segmentation import GM_analysis as gma

from skimage.morphology import skeletonize

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-gt", "--gt_folder", help="Input folder with all scans and ground truth masks")
    parser.add_argument("-sct", "--sct_folder", help="Input folder with all SCT generated masks")
    parser.add_argument("-mt", "--mask_tempaltes", help="Pickle file with template file")
    parser.add_argument("-o", "--output", help="Output folder for the SCT GM mask scores")
    args = parser.parse_args()

    train_file_tree = dm.load_files(args.gt_folder, args.sct_folder)

    avg_mask_file = open(args.mask_tempaltes, "rb")
    avg_masks = pkl.load(avg_mask_file)
    avg_mask_file.close()
    crop_size = avg_masks['site1'].shape[:1]

    donnees = {
        'SITE': [],
        'SCAN': [],
        'SLICE': [],
        'LEVEL': [],
        'DENSITY': [],
        'TEXTURE': [],
        'SHAPE': [],
    }

    stats = {
        'site1':{},
        'site2':{},
        'site3':{},
        'site4':{},
    }

    start_time = time.time()

    for site in train_file_tree.keys():
        dens = []
        text = []
        shap = []
        for sc in train_file_tree[site].keys():

            slice_densities = gma.get_sct_gm_densities(train_file_tree[site][sc])

            for sl in range(train_file_tree[site][sc]['slice_nbr']):

                sct_gm_skeleton = gma.get_cropped_sct_gm_skeleton(train_file_tree[site][sc], sl, crop_size)

                if len(sct_gm_skeleton) > 1:
                    donnees['SITE'].append(int(site[-1]))
                    donnees['SCAN'].append(int(sc[-2:]))
                    donnees['SLICE'].append(sl)
                    try:
                        donnees['LEVEL'].append(int(train_file_tree[site][sc]['levels'][sl][0]))
                    except:
                        donnees['LEVEL'].append(0)
                    donnees['DENSITY'].append(np.mean(slice_densities[sl]))
                    dens.append(donnees['DENSITY'][-1])
                    donnees['TEXTURE'].append(np.std(slice_densities[sl]))
                    text.append(donnees['TEXTURE'][-1])
                    donnees['SHAPE'].append(gma.general_hausdorff_distance(sct_gm_skeleton, avg_masks[site]))
                    shap.append(donnees['SHAPE'][-1])

        stats[site] = {
            'density': {
                'mean': np.mean(dens),
                'std': np.std(dens)
            },
            'texture': {
                'mean': np.mean(text),
                'std': np.std(text)
            },
            'shape': {
                'mean': np.mean(shap),
                'std': np.std(shap)
            }
        }

    df = pd.DataFrame(donnees, columns=list(donnees.keys()))

    df.to_pickle(args.output+"GM_features.pkl")

    end_time = time.time()
    dur = end_time-start_time

    output_file = open(args.output+"GM_feature_stats.pkl", "wb")
    pkl.dump(stats, output_file)
    output_file.close()

    print("All done !")
    print("Total duration : "+str(int(dur//60))+"min and "+str(int(dur%60))+"s")

    print("Saved feature data at : "+args.output+"GM_features.pkl")
    print("Saved feature stats at : "+args.output+"GM_feature_stats.pkl")


if __name__ == "__main__":
    main()