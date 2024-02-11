import datetime
import numpy as np
import scipy
from pathlib import Path
import cv2
import h5py
import torch
from matplotlib import pyplot as plt


CPU = False
device = torch.device('cuda')
factor = 1
clahe_clip_limit = 40
clahe_grid_size = 8


def supersample(matrix: np.ndarray) -> np.ndarray:
    matrix2 = np.empty(shape=(matrix.shape[0] * factor, matrix.shape[1] * factor), dtype=matrix.dtype)
    for i in range(factor):
        for j in range(factor):
            matrix2[i::factor, j::factor] = matrix
    return matrix2


def find_flow_for_cell(mask: np.ndarray, centroid: np.ndarray) -> np.ndarray:
    iterations = 2 * factor * max(mask.shape)
    kernel = np.ones(shape=(3, 3)) * (1. / 9)
    arr = supersample(np.zeros_like(mask))
    mask = supersample(mask)
    centroid *= factor

    if CPU:
        for _ in range(iterations):
            arr[centroid[0], centroid[1]] = 1.0
            arr = scipy.signal.convolve2d(arr, kernel, mode='same', boundary='fill', fillvalue=0)
            arr *= mask
        arr = np.log1p(arr)
    else:
        arr_gpu = torch.from_numpy(arr).resize(1, arr.shape[0], arr.shape[1]).to(device)
        mask_gpu = torch.from_numpy(mask).resize(1, mask.shape[0], mask.shape[1]).to(device)
        centroid_gpu = torch.from_numpy(centroid.astype(int)).to(device)
        kernel_gpu = torch.from_numpy(kernel).resize(1, 1, kernel.shape[0], kernel.shape[1]).to(device)
        for _ in range(iterations):
            arr_gpu[:, centroid_gpu[0], centroid_gpu[1]] = 1.0
            arr_gpu = torch.nn.functional.conv2d(arr_gpu, kernel_gpu, padding='same')
            arr_gpu *= mask_gpu
        del mask_gpu, centroid_gpu, kernel_gpu
        arr = torch.log1p(arr_gpu).cpu().squeeze()
        del arr_gpu

    dAdx = scipy.ndimage.sobel(arr, mode='constant', axis=0) * mask
    dAdy = scipy.ndimage.sobel(arr, mode='constant', axis=1) * mask

    mags = np.sqrt(dAdx**2 + dAdy**2)
    dAdx /= mags
    dAdy /= mags
    angles = np.arctan2(dAdx, dAdy)

    return angles


def find_centroid(mask: np.ndarray) -> np.ndarray:
    kernel = np.ones(shape=(3, 3)) * (1. / 9)
    arr = mask.copy()
    iterations = max(mask.shape)
    for _ in range(iterations):
        arr = scipy.signal.convolve2d(arr, kernel, mode='same', boundary='fill', fillvalue=0)
        arr *= mask
    return np.array(np.unravel_index(np.argmax(arr), arr.shape))


def find_flow(image: np.ndarray) -> np.ndarray:
    objects = scipy.ndimage.find_objects(image)
    slices = []
    cell_flows = []
    for index, cell_slice in enumerate(objects):
        sx, sy = cell_slice
        upsampled_slice = slice(sx.start * factor, sx.stop * factor), slice(sy.start * factor, sy.stop * factor)
        slices.append(upsampled_slice)
        lx, ly = sx.stop - sx.start, sy.stop - sy.start
        mask = np.zeros(shape=(lx, ly), dtype=np.float64)
        mask[image[sx, sy] == (index + 1)] = 1.0

        centroid = find_centroid(mask)

        cell_flow = find_flow_for_cell(mask, centroid)
        cell_flows.append(cell_flow)

    full_flow = np.full(shape=(image.shape[0] * factor, image.shape[1] * factor), fill_value=np.nan, dtype=np.float64)
    for i in range(len(slices)):
        flow = cell_flows[i]
        sl = slices[i]
        full_flow[sl][np.isfinite(flow)] = flow[np.isfinite(flow)]

    #plt.imshow(full_flow, cmap='gist_rainbow')
    #plt.show()

    return full_flow


def flow_polar_to_cartesian(flow: np.ndarray) -> np.ndarray:
    x = np.cos(flow)
    y = np.sin(flow)
    return np.nan_to_num(np.stack([x, y], axis=-1))


def find_centroids(label: np.ndarray) -> np.ndarray:
    objects = scipy.ndimage.find_objects(label)
    result = []
    for i, s in enumerate(objects):
        sx, sy = s
        lx, ly = sx.stop - sx.start, sy.stop - sy.start
        mask = np.zeros(shape=(lx, ly), dtype=np.float64)
        mask[label[sx, sy] == (i + 1)] = 1.0
        x, y = find_centroid(mask)
        result.append([x + sx.start, y + sy.start])
    return np.array(result)


def apply_clahe(image: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=(clahe_grid_size, clahe_grid_size))
    return clahe.apply(image)


def convert_label(raw_label: np.ndarray) -> (np.ndarray, np.ndarray):
    label_enumerated_cells = np.zeros(shape=raw_label.shape[:-1], dtype=np.uint16)
    colors = np.unique(image_raw.reshape((-1, 3)), axis=0)
    obj_index = 1
    for color in colors[1:]:  # skip black color (background)
        mask = np.zeros_like(label_enumerated_cells, dtype=np.uint8)
        mask[(image_raw == color).all(axis=-1)] = 1
        num_objects, labels = cv2.connectedComponents(mask)
        for n in range(1, num_objects):
            label_enumerated_cells[labels == n] = obj_index
            obj_index += 1
    label_binary = label_enumerated_cells > 0
    return label_binary, label_enumerated_cells


if __name__ == '__main__':
    labels_dir = Path('X:/BA/nntraining/training/datasets/bamicrostructure13/raw/SegmentationObject/')
    inputs_dir = Path('X:/BA/nntraining/training/datasets/bamicrostructure13/inputs/')
    output_file = Path('dataset') / 'vectorized_bams13.h5'

    with (h5py.File(output_file, 'w') as h5file):
        instant = str(datetime.datetime.now())

        inputs = h5file.create_group('inputs')
        inputs.attrs['generated_at'] = instant

        for image_file in inputs_dir.iterdir():
            datapoint_name = image_file.stem.replace(' ', '_')
            image_raw = cv2.imread(str(image_file), cv2.IMREAD_ANYCOLOR).astype(np.uint16)
            image_raw = image_raw[:512,:512]
            image_raw = apply_clahe(image_raw).astype(np.float64)
            image_raw = (image_raw - image_raw.mean()) / image_raw.std()
            inputs.create_dataset(datapoint_name, data=image_raw)

        labels_enumerated = h5file.create_group('labels_enumerated')
        labels_enumerated.attrs['generated_at'] = instant
        labels_binary = h5file.create_group('labels_binary')
        labels_binary.attrs['generated_at'] = instant
        fields = h5file.create_group('flowfields')
        fields.attrs['generated_at'] = instant

        for image_file in labels_dir.iterdir():
            datapoint_name = image_file.stem.replace(' ', '_')
            image_raw = cv2.imread(str(image_file), cv2.IMREAD_ANYCOLOR).astype(np.uint8)
            image_raw = image_raw[:512,:512]
            label_is_cell, label_enumerated = convert_label(image_raw)
            labels_enumerated.create_dataset(datapoint_name, data=label_enumerated)
            labels_binary.create_dataset(datapoint_name, data=label_is_cell)
            flow = find_flow(label_enumerated)
            fields.create_dataset(datapoint_name + '_polar', data=flow)
            flow = flow_polar_to_cartesian(flow)
            fields.create_dataset(datapoint_name + '_vector', data=flow)
            print(f'Processed file {image_file}')
