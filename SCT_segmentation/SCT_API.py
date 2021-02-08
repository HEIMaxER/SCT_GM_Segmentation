import os

def make_gm_seg(scan, output_path):
    sct_seg_command = "sct_deepseg_gm -i {input} -o {output}"

    if not os.path.exists(output_path):
        os.makedirs(output_path )

    output_file = output_path+'-'.join(scan['name'].split(' '))+'-sctseg.nii'
    sct_seg_command = sct_seg_command.format(input=scan['image'], output=output_file)
    if 'sct_seg' in scan.keys():
        if scan['sct_seg'] == output_file:
            print("GM Segmentation already exists")
            return
    elif os.path.isfile(output_file):
        scan['sct_seg'] = output_file
        print("GM Segmentation already exists")
        return

    os.system(sct_seg_command)
    if os.path.isfile(output_file):
        scan['sct_seg'] = output_file
        print("GM Segmentation Successful !")
