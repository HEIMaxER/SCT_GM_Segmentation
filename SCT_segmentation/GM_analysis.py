import numpy as np
import nibabel as nib

from scipy.spatial.distance import directed_hausdorff
from skimage.morphology import skeletonize


def general_hausdorff_distance(u, v):
    u_i = np.transpose(np.nonzero(u))
    v_i = np.transpose(np.nonzero(v))
    return max(directed_hausdorff(u_i, v_i)[0], directed_hausdorff(v_i, u_i)[0])


def dice_score(scan, mask=0):
    mask_path = scan['masks'][mask]
    data = nib.load(mask_path)
    gt_volume = data.get_fdata()
    gm_volume = np.where(gt_volume == 1., 1, 0)

    sct_path = scan['sct_seg']
    data = nib.load(sct_path)
    sct_volume = data.get_fdata()

    dice_scores = []

    for i in range(scan['slice_nbr']):
        gt_slice = gm_volume[:,:,i]
        seg_slice = sct_volume[:,:,i]
        inter = seg_slice[gt_slice==1]
        if np.sum(gt_slice)>0 or np.sum(seg_slice)>0:
            ds = np.sum(inter)*2.0 / (np.sum(gt_slice) + np.sum(seg_slice))
        else:
            ds = 0
        dice_scores.append(ds)
    return dice_scores


def hausdorff_score(scan, mask):
    mask_path = scan['masks'][mask]
    data = nib.load(mask_path)
    gt_volume = data.get_fdata()
    gm_volume = np.where(gt_volume==1., 1, 0)

    sct_path = scan['sct_seg']
    data = nib.load(sct_path)
    sct_volume = data.get_fdata()

    haus_scores = []


    for i in range(scan['slice_nbr']):
        haus_scores.append(general_hausdorff_distance(gm_volume[:,:,i], sct_volume[:,:,i]))

    return haus_scores


def mask_cross_dice_score(scan, mask1=0, mask2=1):
    mask_path = scan['masks'][mask1]
    data = nib.load(mask_path)
    gt_volume = data.get_fdata()
    m1_volume = np.where(gt_volume == 1., 1, 0)

    mask_path = scan['masks'][mask2]
    data = nib.load(mask_path)
    gt_volume = data.get_fdata()
    m2_volume = np.where(gt_volume == 1., 1, 0)

    dice_scores = []

    for i in range(scan['slice_nbr']):
        m1_slice = m1_volume[:,:,i]
        m2_slice = m2_volume[:,:,i]
        inter = m1_slice[m2_slice==1]
        if np.sum(m1_slice)>0 or np.sum(m2_slice)>0:
            ds = np.sum(inter)*2.0 / (np.sum(m1_slice) + np.sum(m2_slice))
        else:
            ds = 0
        dice_scores.append(ds)
    return dice_scores


def mask_cross_hausdorff_score(scan, mask1=0, mask2=1):
    mask_path = scan['masks'][mask1]
    data = nib.load(mask_path)
    gt_volume = data.get_fdata()
    m1_volume = np.where(gt_volume==1., 1, 0)

    mask_path = scan['masks'][mask2]
    data = nib.load(mask_path)
    gt_volume = data.get_fdata()
    m2_volume = np.where(gt_volume == 1., 1, 0)

    haus_scores = []


    for i in range(scan['slice_nbr']):
        haus_scores.append(general_hausdorff_distance(m1_volume[:,:,i], m2_volume[:,:,i]))

    return haus_scores


def get_sct_gm_densities(scan):
    img_path = scan['image']
    data = nib.load(img_path)
    irm_volume = data.get_fdata()

    sct_mask_path = scan['sct_seg']
    data = nib.load(sct_mask_path)
    sct_volume = data.get_fdata()

    slice_densities = []
    for i in range(scan['slice_nbr']):
        irm_scile = irm_volume[:,:,i]
        sct_mask = sct_volume[:,:,i]
        slice_densities.append(irm_scile[sct_mask == 1])

    return slice_densities

def get_cropped_gm_mask(scan, sl, mask, crop_size):
    mask_path = scan['masks'][mask]
    data = nib.load(mask_path)
    mask = data.get_fdata()[:, :, sl]

    ids_masks = np.nonzero(mask)
    if len(ids_masks[0]) > 0:
        x_min = min(ids_masks[0])
        x_max = max(ids_masks[0])
        y_min = min(ids_masks[1])
        y_max = max(ids_masks[1])


        xpad = crop_size[0] - (x_max - x_min)
        ypad = crop_size[1] - (y_max - y_min)

        xmin_pad = xpad // 2
        xmax_pad = xpad - xmin_pad
        ymin_pad = ypad // 2
        ymax_pad = ypad - ymin_pad

        cropped_mask = mask[x_min - xmin_pad:x_max + xmax_pad, y_min - ymin_pad:y_max + ymax_pad]

        return cropped_mask

    else:
        return np.zeros(1)


def get_cropped_sct_gm(scan, sl, crop_size):
    sct_mask_path = scan['sct_seg']
    data = nib.load(sct_mask_path)
    sct_mask = data.get_fdata()[:, :, sl]

    ids_masks = np.nonzero(sct_mask)
    if len(ids_masks[0]) > 0:
        x_min = min(ids_masks[0])
        x_max = max(ids_masks[0])
        y_min = min(ids_masks[1])
        y_max = max(ids_masks[1])

        xpad = crop_size[0] - (x_max - x_min)
        ypad = crop_size[0] - (y_max - y_min)

        xmin_pad = xpad // 2
        xmax_pad = xpad - xmin_pad
        ymin_pad = ypad // 2
        ymax_pad = ypad - ymin_pad

        cropped_sct_mask = sct_mask[x_min - xmin_pad:x_max + xmax_pad, y_min - ymin_pad:y_max + ymax_pad]

        return cropped_sct_mask

    else:
        return np.zeros(1)


def get_cropped_sct_gm_skeleton(scan, sl, crop_size):
    sct_mask_path = scan['sct_seg']
    data = nib.load(sct_mask_path)
    sct_mask = skeletonize(data.get_fdata()[:, :, sl])

    ids_masks = np.nonzero(sct_mask)
    if len(ids_masks[0]) > 0:
        x_min = min(ids_masks[0])
        x_max = max(ids_masks[0])
        y_min = min(ids_masks[1])
        y_max = max(ids_masks[1])

        xpad = crop_size[0]-(x_max-x_min)
        ypad = crop_size[0]-(y_max-y_min)

        xmin_pad = xpad//2
        xmax_pad = xpad-xmin_pad
        ymin_pad = ypad//2
        ymax_pad = ypad-ymin_pad

        cropped_sct_mask = sct_mask[x_min-xmin_pad:x_max+xmax_pad, y_min-ymin_pad:y_max+ymax_pad]

        return cropped_sct_mask

    else:
        return np.zeros(1)