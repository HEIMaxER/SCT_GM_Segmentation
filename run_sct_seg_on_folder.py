import argparse
import time
from SCT_segmentation import data as dm
from SCT_segmentation import SCT_API as sct

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="Input folder with all scans to segment")
    parser.add_argument("-o", "--output", help="Output folder for the SCT GM masks")
    args = parser.parse_args()

    file_tree = dm.load_files(args.input)

    total_slices = 0
    start_time = time.time()

    for site in file_tree.keys():
        print("Stating GM segmentation for "+site+" images")
        sc_nbr = len(file_tree[site].keys())
        k=1
        for sc in file_tree[site].keys():
            print("Segmenting scan "+str(k)+" out of "+str(sc_nbr))
            total_slices += file_tree[site][sc]['slice_nbr']
            sct.make_gm_seg(file_tree[site][sc], args.output)
            k+=1
    end_time = time.time()
    dur = end_time-start_time
    print("All done !")
    print("Total duration : "+str(int(dur//60))+"min and "+str(int(dur%60))+"s")
    print(str(total_slices)+" slices segmented")
    print("Average segmentation time per slice : "+str(int(dur//total_slices))+"s")


if __name__ == "__main__":
    main()