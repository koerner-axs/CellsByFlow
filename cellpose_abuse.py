from cellpose import plot, dynamics, utils
import numpy as np
import cv2
from pathlib import Path
from matplotlib import pyplot as plt
import scipy


def _extend_centers_convolution(T: np.ndarray, object_y, object_x, ymed, xmed):
    niter = 2 * (T.shape[0] + T.shape[1])
    kernel_size = 3
    kernel = (kernel_size ** -2) * np.ones((kernel_size, kernel_size), dtype=np.float64)
    for iteration in range(niter):
        T[ymed, xmed] += 1
        #T[object_y, object_x] = scipy.ndimage.correlate(T, kernel, mode='constant')[object_y, object_x]
        T[object_y, object_x] = scipy.ndimage.gaussian_filter(T, sigma=3.0, mode='constant')[object_y, object_x]
        #T = scipy.ndimage.gaussian_filter(T, sigma=3.0, mode='constant')
    return T


def masks_to_flows_cpu(masks, device=None):
    total_size_y, total_size_x = masks.shape
    flow_field = np.zeros((2, total_size_y, total_size_x), np.float64)
    distance_field = np.zeros((total_size_y, total_size_x), np.float64)

    nmask = masks.max()
    slices = scipy.ndimage.find_objects(masks)

    # Hack because my objects are not generally circular.
    # Computes the median box size (counting pixels per object, taking the sqrt).
    # Factor is the hack.
    diameter = utils.diameters(masks)[0] * 3.0

    T_cum = np.zeros((total_size_y, total_size_x), np.float64)

    s2 = (.15 * diameter) ** 2
    for object_index, object_slice in enumerate(slices):
        if object_slice is not None:
            slice_y, slice_x = object_slice
            size_y, size_x = slice_y.stop - slice_y.start + 1, slice_x.stop - slice_x.start + 1

            # Index arrays, work like relative coordinates, pretty sure about that
            object_y, object_x = np.nonzero(masks[slice_y, slice_x] == (object_index + 1))
            object_y = object_y.astype(np.int32) + 1  # Add 1 for easy padding later on
            object_x = object_x.astype(np.int32) + 1  # in the diffusion step.
            # Find rough median point first.
            ymed = np.median(object_y)
            xmed = np.median(object_x)
            # Then correct to the nearest point of the object.
            imin = np.argmin((object_x - xmed) ** 2 + (object_y - ymed) ** 2)
            ymed = object_y[imin]
            xmed = object_x[imin]

            # The reference point is (ymed, xmed). Compute distances from it.
            squared_dist_to_ref = (object_x - xmed) ** 2 + (object_y - ymed) ** 2
            dist = np.exp(-squared_dist_to_ref / s2)
            distance_field[slice_y.start + object_y - 1, slice_x.start + object_x - 1] = dist

            num_diffusion_steps = 2 * np.int32(np.ptp(object_x) + np.ptp(object_y))
            T = np.zeros((size_y + 2, size_x + 2), np.float64)
            T = _extend_centers_convolution(T, object_y, object_x, ymed, xmed)
            # T = _extend_centers_gpu(T)

            # T[(object_y+1)*size_x + object_x+1] += 1
            #T[object_y+1, object_x+1] = np.log(1.+T[object_y+1, object_x+1])

            # Differentiate, ignore factor of 2.
            #dy = T[object_y + 1, object_x] - T[object_y - 1, object_x]
            #dx = T[object_y, object_x + 1] - T[object_y, object_x - 1]
            dTdy = scipy.ndimage.sobel(T, axis=0, mode='constant')
            dTdx = scipy.ndimage.sobel(T, axis=1, mode='constant')
            dTdy = dTdy[object_y, object_x]
            dTdx = dTdx[object_y, object_x]

            flow_field[:, slice_y.start + object_y - 1, slice_x.start + object_x - 1] = np.stack((dTdy, dTdx))

            T_cum[slice_y.start + object_y - 1, slice_x.start + object_x - 1] = T[object_y, object_x]

    flow_field /= (1e-20 + (flow_field ** 2).sum(axis=0) ** 0.5)

    from matplotlib import pyplot as plt
    T_cum = np.log(T_cum)
    plt.imshow(T_cum)
    plt.show()

    return flow_field, distance_field


input_file = Path('./input_15_BD out of plane 7.png')
label_file = Path('./label_15_BD out of plane 7.png')
raw_input = cv2.imread(str(input_file), cv2.IMREAD_GRAYSCALE)
raw_label = cv2.imread(str(label_file), cv2.IMREAD_COLOR)

# Find unique colors
colors = np.unique(np.reshape(raw_label, newshape=(-1, 3)), axis=0)

# Convert to int repr
int_label = np.zeros(shape=raw_label.shape[:2], dtype=np.uint32)
for color_index, color in enumerate(colors):
    occurence = (raw_label == color).all(axis=-1)
    # Decide if cell is connected, if not show warning and ignore this cell
    # seen = np.zeros_like(occurence, dtype=np.bool)
    # start = np.where(occurence)[0]
    # stack = [start]
    # while len(stack) > 0:
    #    current = stack.pop(-1)
    #    if not seen[current]:
    #        seen[current] = True
    #        if

    int_label[occurence] = color_index

# Apply diffusion algorithm to turn masks into flows
mu, mu_c = masks_to_flows_cpu(int_label)

# plt.imshow(mu_c)
# plt.show()
# plt.imshow(mu[0])
# plt.show()
# plt.imshow(mu[1])
# plt.show()
mu_hsv = plot.dx_to_circ(mu)
# mu_ext = np.empty(shape=(3, mu.shape[1], mu.shape[2]), dtype=np.float32)
# mu_ext[0:2] = mu
# mu_ext[2] = (raw_label != np.array([0.0, 0.0, 0.0])).any(axis=-1)


plt.imshow(np.sqrt(np.sum(mu**2,axis=0)))
plt.show()
plt.imshow(mu[0])
plt.show()
plt.imshow(mu[1])
plt.show()
plt.imshow(mu_hsv)
plt.show()

# fig = plt.figure()
# plot.show_segmentation(fig, raw_input, int_label, mu_hsv)
# plt.show()
