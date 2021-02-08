import argparse
import time
import pickle
from SCT_segmentation import data as dm
from SCT_segmentation import GM_analysis as gma

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-gt", "--gt_folder", help="Input folder with all scans and ground truth masks")
    parser.add_argument("-sct", "--sct_folder", help="Input folder with all SCT generated masks")
    parser.add_argument("-o", "--output", help="Output folder for the SCT GM mask scores")
    args = parser.parse_args()

    file_tree = dm.load_files(args.gt_folder, args.sct_folder)

    score_tree = {
        'maskr1':{},
        'maskr2':{},
        'maskr3':{},
        'maskr4':{}
    }
    masks = list(score_tree.keys())
    ds = []
    hs = []
    start_time = time.time()

    for i in range(len(masks)):
        print("Getting scores for " + masks[i])
        for site in file_tree.keys():
            print("Getting scores for "+site+" images")
            score_tree[masks[i]][site] = {}
            sc_nbr = len(file_tree[site].keys())
            k=1
            for sc in file_tree[site].keys():
                score_tree[masks[i]][site][sc] = {}
                print("Scoring segmentation of scan "+str(k)+" out of "+str(sc_nbr))
                dice_scores = gma.dice_score(file_tree[site][sc], i)
                n = sum(x > 0 for x in dice_scores)
                haus_scores = gma.hausdorff_score(file_tree[site][sc], i)
                avg_dice_score = sum(dice_scores)/n
                avg_haus_score = sum(haus_scores)/n
                score_tree[masks[i]][site][sc]['dice_score'] = avg_dice_score
                score_tree[masks[i]][site][sc]['haus_score'] = avg_haus_score
                ds.append(avg_dice_score)
                hs.append(avg_haus_score)

                k+=1
    end_time = time.time()
    dur = end_time-start_time

    output_file = open(args.output+"sct_seg_scores.pkl", "wb")
    pickle.dump(score_tree, output_file)
    output_file.close()

    print("All done !")
    print("Total duration : "+str(int(dur//60))+"min and "+str(int(dur%60))+"s")
    print("Average dice score : "+str(sum(ds)/len(ds)))
    print("Average hausdorff distance : "+str(sum(hs)/len(hs)))

    print("Saved scores at : " + args.output + "sct_seg_scores.pkl")

if __name__ == "__main__":
    main()