from skimage.morphology import ball, disk, dilation, binary_erosion, remove_small_objects, erosion, closing, \
    reconstruction, binary_closing
from skimage.measure import label, regionprops
from skimage.filters import roberts
from skimage.segmentation import clear_border
from scipy import ndimage as ndi
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from glob import glob
from functools import lru_cache

from tqdm import tqdm
import pydicom


def get_segmented_lungs(img_dir,  # s1_path, s2_path, s3_path,
                        plot=False, min_lung_area=4000, padding_num=0,
                        mean_begin=20, mean_end=200, judge_threshold=0.001, raise_exception=False
                        ):
    """
    :param img_dir: 输入图片路径
    :param s1_path: 原图保存路径
    :param s2_path: 经过截取的有用部分的图片保存路径
    :param s3_path: 经过填充的图片保存路径
    :param plot: 是否展示图片
    :param min_lung_area: 一个肺部的可能最小面积（作为阈值）
    :param padding_num: 截取部分需要padding的大小
    :return: True or False (img is useful or not)
    """
    if isinstance(img_dir, str):
        src_img = cv2.imread(img_dir)
        img = Image.open(img_dir)
    else:
        src_img = img_dir
        img = Image.fromarray(img_dir.squeeze())
    assert len(src_img.shape) == 3, f'Current shape:{src_img.shape}'
    src_img1 = src_img.copy()
    img = img.convert('RGB')
    img = np.array(img)[:, :, 0]
    img = img.astype(np.int16)

    # print(img.shape)
    if img.mean() < mean_begin or img.mean() > mean_end:
        print(f'Warning: img.mean()={img.mean()}, not between {mean_begin}, {mean_end} ')
        if raise_exception:
            raise Exception(f'img.mean()={img.mean()}, not between {mean_begin}, {mean_end}')
        else:
            return src_img, src_img, 1, 255
    img = img * 3
    img -= 1000
    im = img.copy()
    '''
    This funtion segments the lungs from the given 2D slice.
    '''
    if plot == True:
        f, plots = plt.subplots(8, 1, figsize=(5, 40))
    '''
    Step 1: Convert into a binary image. 
    '''

    thresold_binary = min(-600, np.median(im) + 200)
    # print('mean', np.mean(im), np.std(im), np.median(im), thresold_binary)

    # 数字越小,去除的白色越多
    binary = im < thresold_binary
    if plot == True:
        plots[0].axis('off')
        plots[0].imshow(binary, cmap=plt.cm.bone)
    '''
    Step 2: Remove the blobs connected to the border of the image.
    '''
    cleared = clear_border(binary)
    if plot == True:
        plots[1].axis('off')
        plots[1].imshow(cleared, cmap=plt.cm.bone)
    '''
    Step 3: Label the image.
    '''
    label_image = label(cleared)
    if plot == True:
        plots[2].axis('off')
        plots[2].imshow(label_image, cmap=plt.cm.bone)
    '''
    Step 4: Keep the labels with 2 largest areas.
    '''
    areas = [r.area for r in regionprops(label_image)]
    areas.sort()
    if len(areas) > 2:
        for region in regionprops(label_image):
            if region.area < areas[-2] or region.area < min_lung_area:
                for coordinates in region.coords:
                    label_image[coordinates[0], coordinates[1]] = 0
    binary = label_image > 0
    if plot == True:
        plots[3].axis('off')
        plots[3].imshow(binary, cmap=plt.cm.bone)
    '''
    Step 5: Erosion operation with a disk of radius 2. This operation is 
    seperate the lung nodules attached to the blood vessels.
    '''
    selem = disk(2)
    binary = binary_erosion(binary, selem)
    if plot == True:
        plots[4].axis('off')
        plots[4].imshow(binary, cmap=plt.cm.bone)
    '''
    Step 6: Closure operation with a disk of radius 10. This operation is 
    to keep nodules attached to the lung wall.
    '''
    selem = disk(10)
    binary = binary_closing(binary, selem)
    if plot == True:
        plots[5].axis('off')
        plots[5].imshow(binary, cmap=plt.cm.bone)
    '''
    Step 7: Fill in the small holes inside the binary mask of lungs.
    '''
    edges = roberts(binary)
    binary = ndi.binary_fill_holes(edges)
    if plot == True:
        plots[6].axis('off')
        plots[6].imshow(binary, cmap=plt.cm.bone)
    '''
    Step 8: Superimpose the binary mask on the input image.
    '''
    get_high_vals = binary == 0
    im[get_high_vals] = 0
    if plot == True:
        plots[7].axis('off')
        plots[7].imshow(im, cmap=plt.cm.bone)
    mask = binary.astype(int)

    mask_img = np.zeros(src_img.shape, dtype=src_img.dtype)
    mask_img[mask == 1] = [255, 255, 0]

    # plt.figure(figsize=(12, 12))
    # plt.imshow(mask_img)
    # plt.show()

    if mask.mean() < judge_threshold or mask_img.mean() < 6:

        print(f'Warning: mask.mean={mask.mean()}, not between {judge_threshold} and 6,  ')
        if raise_exception:
            raise Exception(f'mask.mean={mask.mean()}, not between {judge_threshold} and 6, ')
        else:
            return src_img, src_img, 1, 255
    # 找出包围肺部的最小有效矩形
    gray = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    new_img = np.zeros(src_img.shape)
    cv2.drawContours(new_img, contours, -1, (255, 255, 0), thickness=cv2.FILLED)

    # m = new_img[:, :, 0] == 255

    minx, miny, maxx, maxy = src_img.shape[1], src_img.shape[0], 0, 0
    for c in contours:
        if cv2.contourArea(c) > min_lung_area:
            temp_minx = np.min(c[:, 0, 0])
            temp_miny = np.min(c[:, 0, 1])
            temp_maxx = np.max(c[:, 0, 0])
            temp_maxy = np.max(c[:, 0, 1])
            if temp_minx < minx:
                minx = temp_minx
            if temp_miny < miny:
                miny = temp_miny
            if temp_maxx > maxx:
                maxx = temp_maxx
            if temp_maxy > maxy:
                maxy = temp_maxy
    final_img = src_img[miny - padding_num:maxy + padding_num, minx - padding_num:maxx + padding_num]
    final_mask = mask[miny - padding_num:maxy + padding_num, minx - padding_num:maxx + padding_num]
    final_img[final_mask == 0] = [0, 0, 0]

    final_crop = src_img1[miny - padding_num:maxy + padding_num, minx - padding_num:maxx + padding_num]

    # plt.figure(figsize=(9, 9))
    # plt.imshow(final_img, cmap='gray')
    # plt.show()

    # 填充图片
    final_mask = np.stack([final_mask] * 3, 2)

    background = entity_filling(final_img, final_mask)

    filled_img = np.where(final_mask == 1, final_img, background)
    filled_img = np.array(filled_img, dtype=np.int16)

    # plt.figure(figsize=(12, 12))
    # plt.imshow(background)
    # plt.show()
    cut_per = (final_img.shape[0] * final_img.shape[1]) / (src_img.shape[0] * src_img.shape[1])

    img_mean = np.mean(final_img[:, :, 0])
    print(f'cut {cut_per:.2f}, mean:{img_mean}, from {final_img.shape}')
    return final_img, final_crop, cut_per, img_mean


# def lung_load_fn(path):
#     final_img, final_crop, _ = get_segmented_lungs(path,False, mean_begin=0, mean_end =255)
#     #print(type(final_img), path)
#     final_img = Image.fromarray(final_img)
#     return final_img

# =============================================================================
#     shutil.copy(img_dir, s1_path)
#     cv2.imwrite(s2_path, final_img)
#     cv2.imwrite(s3_path, filled_img)
#     return True
# =============================================================================

def entity_filling(src_img, mask):
    background = np.zeros_like(src_img)
    for i in range(40):
        ix = np.random.randint(0, src_img.shape[0])
        if np.random.randint(2) == 0:
            new_img = np.concatenate(
                [np.zeros((src_img.shape[0] - ix, src_img.shape[1], 3)), src_img[:ix, :]], 0)
            new_mask = np.concatenate(
                [np.zeros((src_img.shape[0] - ix, src_img.shape[1], 3)), mask[:ix, :]], 0)
        else:
            new_img = np.concatenate(
                [src_img[ix:, :], np.zeros((ix, src_img.shape[1], 3))], 0)
            new_mask = np.concatenate(
                [mask[ix:, :], np.zeros((ix, src_img.shape[1], 3))], 0)
        ix = np.random.randint(0, src_img.shape[1])
        if np.random.randint(2) == 0:
            new_img = np.concatenate(
                [np.zeros((src_img.shape[0], src_img.shape[1] - ix, 3)), new_img[:, :ix]], 1)
            new_mask = np.concatenate(
                [np.zeros((src_img.shape[0], src_img.shape[1] - ix, 3)), new_mask[:, :ix]], 1)
        else:
            new_img = np.concatenate(
                [new_img[:, ix:], np.zeros((src_img.shape[0], ix, 3))], 1)
            new_mask = np.concatenate(
                [new_mask[:, ix:], np.zeros((src_img.shape[0], ix, 3))], 1)

        background = np.where(new_mask == 1, new_img, background)
    return background


def cut_siglefile(file, input, output, reuse=True):
    try:
        outfile = file.replace(input, output)
        if os.path.exists(outfile) and reuse:
            return True

        npz_obj = np.load(file)
        npz_array = npz_obj.f.arr_0
        os.makedirs(os.path.dirname(outfile), exist_ok=True)
        for sn, slice in enumerate(npz_array):
            png = dicom2png(slice)
            png = hist_match(png)

            img, _, per, img_avg = get_segmented_lungs(png, False)
            if 0.2 <= per <= 0.5 and img_avg > 35:
                img = cv2.resize(img, (448, 448))
                print(f'File save to:{outfile}')
                plt.imsave(f'{outfile}_{npz_array.shape[0]:03}_{sn:03}_{per:.2f}_{int(img_avg)}.png', img)

    except Exception as e:
        print(e)


def cut(input='/share/data/lung/lung_ct_npy_v2', output='/share/data1/lung/lung_img_output_v2', reuse = True):
    from functools import partial
    tmp_fn = partial(cut_siglefile, input=input, output=output, reuse = reuse)
    from multiprocessing import Pool
    todo = tqdm(glob(f'{input}/**/*.npz', recursive=True))

    Pool(10).map(tmp_fn, todo)


def dicom2png(input, pixel_range=3000):
    if isinstance(input, str):
        ds = pydicom.dcmread(input, force=True)
        ds.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
        img = ds.pixel_array
    else:
        img = input
    img = img.clip(img.max() - pixel_range, img.max())
    img = img - img.min()
    img = (np.maximum(img, 0) / img.max()) * 255.0
    img = img.astype(np.uint8)
    img = Image.fromarray(img)
    img = img.convert('RGB')
    return np.array(img)


@lru_cache()
def get_ref(index):
    path = list(glob('./input/**/*.jpg'))[index]
    refer = cv2.imread(path)
    return refer


def hist_match(image, image_index=0):

    from skimage import data
    from skimage import exposure
    from skimage.exposure import match_histograms
    reference = get_ref(image_index)
    matched = match_histograms(image, reference, multichannel=True)
    return matched

if __name__ == '__main__':
    """"
    nohup python -u  preprocessing/extrac_lung.py >> cut.log 2>&1 & 
    """
    cut(reuse=False)


