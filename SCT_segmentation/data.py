import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

from SCT_segmentation import GM_analysis as gma
from skimage.morphology import skeletonize


def load_files(img_path, mask_path=None):
    train_data_file_names = os.listdir(img_path)[1:]
    file_tree= {}

    for fn in train_data_file_names:
        split_fn = fn.split('-')

        if split_fn[0] not in file_tree.keys():
            file_tree[split_fn[0]] = {}

        if split_fn[1] not in file_tree[split_fn[0]].keys():
            file_tree[split_fn[0]][split_fn[1]] = {
                'name': split_fn[0]+' '+split_fn[1],
                'image': '',
                'slice_nbr': 0,
                'levels': [],
                'masks': []
            }

        if 'image' in split_fn[2]:
            file_tree[split_fn[0]][split_fn[1]]['image'] = img_path+fn
            data = nib.load(file_tree[split_fn[0]][split_fn[1]]['image'])
            slices = data.get_fdata()
            file_tree[split_fn[0]][split_fn[1]]['slice_nbr'] = slices.shape[-1]
            file_tree[split_fn[0]][split_fn[1]]['levels'] = ['' for i in range(file_tree[split_fn[0]][split_fn[1]]['slice_nbr'])]
        elif 'mask' in split_fn[2]:
            file_tree[split_fn[0]][split_fn[1]]['masks'].append(img_path+fn)
        elif 'levels' in split_fn[2]:
            with open(img_path+fn) as f:
                lines = f.readlines()
            for l in lines[1:]:
                if l != '':
                    l = l.strip('\n')
                    l = l.strip(' ')

                    slice_id = int(l.split(', ')[0])

                    if slice_id < file_tree[split_fn[0]][split_fn[1]]['slice_nbr']:
                        file_tree[split_fn[0]][split_fn[1]]['levels'][slice_id] = l.split(', ')[-1]

    if mask_path:
        mask_file_names = os.listdir(mask_path)
        for fn in mask_file_names:
            split_fn = fn.split('-')

            if split_fn[0] not in file_tree.keys():
                file_tree[split_fn[0]] = {}

            if split_fn[1] not in file_tree[split_fn[0]].keys():
                file_tree[split_fn[0]][split_fn[1]] = {
                    'name': split_fn[0] + ' ' + split_fn[1],
                    'image': '',
                    'slice_nbr': 0,
                    'levels': [],
                    'masks': []
                }

            if 'sctseg' in split_fn[2]:
                file_tree[split_fn[0]][split_fn[1]]['sct_seg'] = mask_path + fn

    return file_tree

def load_sct_files(sct_path):
    mask_file_names = os.listdir(sct_path)
    file_tree = {}
    for fn in mask_file_names:
        split_fn = fn.split('-')

        if split_fn[0] not in file_tree.keys():
            file_tree[split_fn[0]] = {}

        if split_fn[1] not in file_tree[split_fn[0]].keys():
            file_tree[split_fn[0]][split_fn[1]] = {
                'name': split_fn[0] + ' ' + split_fn[1],
                'slice_nbr': 0,
            }

        if 'sctseg' in split_fn[2]:
            file_tree[split_fn[0]][split_fn[1]]['sct_seg'] = sct_path + fn
            data = nib.load(file_tree[split_fn[0]][split_fn[1]]['sct_seg'])
            slices = data.get_fdata()
            file_tree[split_fn[0]][split_fn[1]]['slice_nbr'] = slices.shape[-1]

    return file_tree


def load_training_files(train_path):
    train_data_file_names = os.listdir(train_path)[1:]
    file_tree= {}

    for fn in train_data_file_names:
        split_fn = fn.split('-')

        if split_fn[0] not in file_tree.keys():
            file_tree[split_fn[0]] = {}

        if split_fn[1] not in file_tree[split_fn[0]].keys():
            file_tree[split_fn[0]][split_fn[1]] = {
                'name': split_fn[0]+' '+split_fn[1],
                'image': '',
                'slice_nbr': 0,
                'levels': [],
                'masks': []
            }

        if 'image' in split_fn[2]:
            file_tree[split_fn[0]][split_fn[1]]['image'] = train_path+fn
            data = nib.load(file_tree[split_fn[0]][split_fn[1]]['image'])
            slices = data.get_fdata()
            file_tree[split_fn[0]][split_fn[1]]['slice_nbr'] = slices.shape[-1]
            file_tree[split_fn[0]][split_fn[1]]['levels'] = ['' for i in
                                                             range(file_tree[split_fn[0]][split_fn[1]]['slice_nbr'])]
        elif 'mask' in split_fn[2]:
            file_tree[split_fn[0]][split_fn[1]]['masks'].append(train_path+fn)
        elif 'levels' in split_fn[2]:
            with open(train_path + fn) as f:
                lines = f.readlines()
            for l in lines[1:]:
                if l != '':
                    l = l.strip('\n')
                    l = l.strip(' ')

                    slice_id = int(l.split(', ')[0])

                    if slice_id < file_tree[split_fn[0]][split_fn[1]]['slice_nbr']:
                        file_tree[split_fn[0]][split_fn[1]]['levels'][slice_id] = l.split(', ')[-1]
    return file_tree


def plot_slices(scan, levels=None):
    img_path = scan['image']
    data = nib.load(img_path)
    slices = data.get_fdata()
    if levels:
        ids = []
        for i in range(scan['slice_nbr']):
            k = 0
            while i not in ids and k <= len(levels)-1:
                if str(levels[k]) in scan['levels'][i]:
                    ids.append(i)
                k+=1
        slice_nbr = len(ids)
    else:
        slice_nbr = scan['slice_nbr']
        ids = range(slice_nbr)

    if slice_nbr > 4:
        if slice_nbr % 4 != 0:
            fig, axs = plt.subplots((slice_nbr//4)+1, 4, figsize=(8,8))
        else:
            fig, axs = plt.subplots((slice_nbr//4), 4, figsize=(8,8))
        fig.suptitle(scan['name'])
        for i in range(len(ids)):
            axs[i//4, i%4].imshow(slices[:, :, ids[i]], cmap='gray')
            axs[i//4, i%4].axis('off')
    else:
        fig, axs = plt.subplots(1, slice_nbr, figsize=(8,8))
        fig.suptitle(scan['name'], y=.8)
        for i in range(len(ids)):
            axs[i].imshow(slices[:, :, ids[i]], cmap='gray')
            axs[i].axis('off')


def plot_masks(scan, masks=None, levels=None):
    mask_paths = scan['masks']
    if not masks:
        masks = range(len(mask_paths))

    mask_nbr = len(masks)

    if levels:
        ids = []
        for i in range(scan['slice_nbr']):
            k = 0
            while i not in ids and k <= len(levels)-1:
                if str(levels[k]) in scan['levels'][i]:
                    ids.append(i)
                k+=1
        slice_nbr = len(ids)
    else:
        slice_nbr = scan['slice_nbr']
        ids = range(slice_nbr)


    if mask_nbr > 1:
        fig, axs = plt.subplots(mask_nbr, slice_nbr)
        fig.suptitle(scan['name'], y=.95)
        for i in range(mask_nbr):
            mask_path = mask_paths[masks[i]]
            data = nib.load(mask_path)
            slices = data.get_fdata()
            for j in range(slice_nbr):
                axs[i, j].imshow(slices[:, :, ids[j]])
                axs[i, j].set_title('Mask '+str(masks[i]+1)+', Level ' + scan['levels'][ids[j]], fontsize=10)
                axs[i, j].axis('off')
    else:
        fig, axs = plt.subplots(mask_nbr, slice_nbr)
        fig.suptitle(scan['name'], y=.8)
        mask_path = mask_paths[masks[0]]
        data = nib.load(mask_path)
        slices = data.get_fdata()
        for j in range(slice_nbr):
            axs[j].imshow(slices[:, :, ids[j]])
            axs[j].set_title('Mask ' + str(masks[0]) + ', Level ' + scan['levels'][ids[j]], fontsize=10)
            axs[j].axis('off')


def plot_gm_ground_truth(scan, mask=0, slice=0):
    img_path = scan['image']
    data = nib.load(img_path)
    img_slice = data.get_fdata()[:,:,slice]
    img_slice /= img_slice.max()
    img = np.asarray(np.dstack((img_slice, img_slice, img_slice)))

    mask_path = scan['masks'][mask]
    data = nib.load(mask_path)
    gt_slice = data.get_fdata()[:,:,slice]
    gt_img = np.where(np.stack((gt_slice,)*3, axis=-1)==(1., 1., 1.),np.stack((np.zeros(gt_slice.shape), gt_slice,np.zeros(gt_slice.shape)), axis=-1), img)

    fig, axs = plt.subplots(1, 2)
    fig.suptitle(scan['name']+' level '+scan['levels'][slice])

    axs[0].imshow(img)
    axs[0].set_title('Raw image level '+scan['levels'][slice], fontsize=20)
    axs[0].axis('off')

    axs[1].imshow(gt_img)
    axs[1].set_title('Gray matter Ground Truth '+str(mask+1), fontsize=20)
    axs[1].axis('off')


def plot_sct_seg_and_gt(scan, mask=0, slice=0):

    if not scan['sct_seg']:
        print("No SCT segmentation found !")
        return

    img_path = scan['image']
    data = nib.load(img_path)
    img_slice = data.get_fdata()[:,:,slice]
    img_slice /= img_slice.max()
    img = np.asarray(np.dstack((img_slice, img_slice, img_slice)))

    mask_path = scan['masks'][mask]
    data = nib.load(mask_path)
    gt_slice = data.get_fdata()[:,:,slice]
    gt_img = np.where(np.stack((gt_slice,)*3, axis=-1)==(1., 1., 1.),np.stack((np.zeros(gt_slice.shape), gt_slice,np.zeros(gt_slice.shape)), axis=-1), img)

    mask_path = scan['sct_seg']
    data = nib.load(mask_path)
    sct_slice = data.get_fdata()[:, :, slice]
    sct_img = np.where(np.stack((sct_slice,) * 3, axis=-1) == (1., 1., 1.),
                      np.stack((sct_slice, np.zeros(sct_slice.shape),  np.zeros(sct_slice.shape)), axis=-1), img)

    fig, axs = plt.subplots(1, 2)
    fig.suptitle(scan['name']+' level '+scan['levels'][slice])

    axs[0].imshow(gt_img)
    axs[0].set_title('Gray matter Ground Truth '+str(mask+1), fontsize=20)
    axs[0].axis('off')

    axs[1].imshow(sct_img)
    axs[1].set_title('SCT Segmentation', fontsize=20)
    axs[1].axis('off')


def plot_sklt_sct_masks(scan, mask=0, slice=0):

    if not scan['sct_seg']:
        print("No SCT segmentation found !")
        return

    img_path = scan['image']
    data = nib.load(img_path)
    img_slice = data.get_fdata()[:,:,slice]
    img_slice /= img_slice.max()
    img = np.asarray(np.dstack((img_slice, img_slice, img_slice)))

    mask_path = scan['sct_seg']
    data = nib.load(mask_path)
    sct_slice = data.get_fdata()[:, :, slice]
    sklt_sct_slice = skeletonize(sct_slice)
    sct_img = np.where(np.stack((sklt_sct_slice,) * 3, axis=-1) == (1., 1., 1.),
                       np.stack((sklt_sct_slice, np.zeros(sklt_sct_slice.shape), np.zeros(sklt_sct_slice.shape)), axis=-1), img)

    mask_path = scan['masks'][mask]
    data = nib.load(mask_path)
    gt_slice = data.get_fdata()[:,:,slice]
    gt_slice = np.where(gt_slice == 1., 1, 0)
    sklt_gt_slice = skeletonize(gt_slice)
    gt_img = np.where(np.stack((sklt_gt_slice,)*3, axis=-1)==(1., 1., 1.),np.stack((np.zeros(sklt_gt_slice.shape), sklt_gt_slice,np.zeros(sklt_gt_slice.shape)), axis=-1), sct_img)

    ids_masks = np.nonzero(sklt_sct_slice)
    x_min = min(ids_masks[0])
    x_max = max(ids_masks[0])
    y_min = min(ids_masks[1])
    y_max = max(ids_masks[1])
    pad = 20

    gt_img = gt_img[x_min-pad:x_max+pad, y_min-pad:y_max+pad]

    plt.imshow(gt_img)
    plt.title('Masques expert '+str(mask+1)+' (vert) et SCT (rouge) skelétisés '+scan['name']+' level '+scan['levels'][slice], fontsize=20)
    plt.show()

    print('Distance de Hausdorff skeletisée : ', gma.general_hausdorff_distance(sklt_sct_slice, sklt_gt_slice))


def plot_sct_masks_overlap(scan, mask=0, slice=0):

    if not scan['sct_seg']:
        print("No SCT segmentation found !")
        return

    img_path = scan['image']
    data = nib.load(img_path)
    img_slice = data.get_fdata()[:,:,slice]
    img_slice /= img_slice.max()
    img = np.asarray(np.dstack((img_slice, img_slice, img_slice)))

    mask_path = scan['masks'][mask]
    data = nib.load(mask_path)
    gt_slice = data.get_fdata()[:, :, slice]
    gt_slice = np.where(gt_slice == 1., 1, 0)
    gt_img = np.where(np.stack((gt_slice,) * 3, axis=-1) == (1., 1., 1.),
                      np.stack((np.zeros(gt_slice.shape), gt_slice, np.zeros(gt_slice.shape)), axis=-1), img)

    mask_path = scan['sct_seg']
    data = nib.load(mask_path)
    sct_slice = data.get_fdata()[:, :, slice]
    sct_img = np.where(np.stack((sct_slice,) * 3, axis=-1) == (1., 1., 1.),
                       np.stack((sct_slice, np.zeros(sct_slice.shape), np.zeros(sct_slice.shape)), axis=-1), gt_img)

    ids_masks = np.nonzero(sct_slice)
    x_min = min(ids_masks[0])
    x_max = max(ids_masks[0])
    y_min = min(ids_masks[1])
    y_max = max(ids_masks[1])
    pad = 20

    sct_img = sct_img[x_min-pad:x_max+pad, y_min-pad:y_max+pad]

    plt.imshow(sct_img)
    plt.title('Masques expert '+str(mask+1)+' (vert) et SCT (rouge) ', fontsize=20)
    plt.show()

    print(scan['name']+' level '+scan['levels'][slice])
    print('Distance de Hausdorff skeletisée : ', gma.general_hausdorff_distance(gt_slice, sct_slice))
    print('Score Dice : ', gma.dice_score(scan, mask)[slice])

def plot_sklt_sct_masks_ref(scan, avg_mask, mask=0, slice=0):

    if not scan['sct_seg']:
        print("No SCT segmentation found !")
        return

    img_path = scan['image']
    data = nib.load(img_path)
    img_slice = data.get_fdata()[:,:,slice]
    print(img_slice.max(), img_slice.min())
    img_slice /= img_slice.max()
    img = np.asarray(np.dstack((img_slice, img_slice, img_slice)))

    mask_path = scan['sct_seg']
    data = nib.load(mask_path)
    sct_slice = data.get_fdata()[:, :, slice]
    sklt_sct_slice = skeletonize(sct_slice)
    sct_img = np.where(np.stack((sklt_sct_slice,) * 3, axis=-1) == (1., 1., 1.),
                       np.stack((sklt_sct_slice, np.zeros(sklt_sct_slice.shape), np.zeros(sklt_sct_slice.shape)), axis=-1), img)

    ids_masks = np.nonzero(sklt_sct_slice)
    x_min = min(ids_masks[0])
    x_max = max(ids_masks[0])
    y_min = min(ids_masks[1])
    y_max = max(ids_masks[1])

    xpad = avg_mask.shape[0] - (x_max - x_min)
    ypad = avg_mask.shape[1] - (y_max - y_min)

    xmin_pad = xpad // 2
    xmax_pad = xpad - xmin_pad
    ymin_pad = ypad // 2
    ymax_pad = ypad - ymin_pad

    sct_img = sct_img[x_min - xmin_pad:x_max + xmax_pad, y_min - ymin_pad:y_max + ymax_pad]

    gt_img = np.where(np.stack((avg_mask,)*3, axis=-1)==(1., 1., 1.),np.stack((np.zeros(avg_mask.shape), avg_mask,np.zeros(avg_mask.shape)), axis=-1), sct_img)

    plt.imshow(gt_img)
    plt.title('Masques expert '+str(mask+1)+' (vert) et SCT (rouge) skelétisés '+scan['name']+' level '+scan['levels'][slice], fontsize=20)
    plt.show()

    print('Distance de Hausdorff skeletisée : ', gma.general_hausdorff_distance(sklt_sct_slice[x_min - xmin_pad:x_max + xmax_pad, y_min - ymin_pad:y_max + ymax_pad], avg_mask))
    ds = gma.get_sct_gm_densities(scan)
    print('Densité :', np.mean(ds[slice]))
    print('Texute :', np.std(ds[slice]))