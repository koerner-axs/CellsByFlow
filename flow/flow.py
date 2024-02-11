import numpy as np
import scipy
from pathlib import Path
import cv2
import h5py
import torch
import numba as nb
from matplotlib import pyplot as plt


uint16max = np.iinfo(np.int16).max
input_file = Path('/dataset/vectorized_bams13.h5')
device = torch.device('cuda')


def make_binary_mask(label: np.ndarray) -> np.ndarray:
    mask = np.zeros_like(label, dtype=np.uint8)
    mask[np.nan_to_num(label) != 0] = 1
    return mask


def decode_polar_flow(polar_flow: np.ndarray) -> np.ndarray:
    flow = np.stack([np.sin(polar_flow), np.cos(polar_flow)], axis=-1)
    flow = np.nan_to_num(flow)

    return flow


def simulate_flow(mask: np.ndarray, flow: np.ndarray, niter: int) -> np.ndarray:
    particles = np.zeros(shape=(*mask.shape, 2), dtype=np.float32)
    for x in range(mask.shape[0]):
        for y in range(mask.shape[1]):
            particles[x,y,:] = (x, y)

    # Normalize to be able to use torch's grid_sample
    shape = np.array(mask.shape)
    particles = particles * 2 / shape - 1
    flow = flow * 2 / shape
    particles_gpu = torch.from_numpy(particles).to(device).unsqueeze(0)
    flow_gpu = torch.from_numpy(flow).to(device).permute(2, 1, 0).unsqueeze(0)

    for _ in range(niter):
        dF = torch.nn.functional.grid_sample(flow_gpu, particles_gpu, align_corners=False)
        for c in range(2):
            particles_gpu[:,:,:,c] = torch.clamp(particles_gpu[:,:,:,c] + dF[:,c,:,:], -1., 1.)

    particles = particles_gpu.cpu().squeeze().numpy()
    del particles_gpu, flow_gpu

    # Renormalize and clamp to image size
    particles = (particles + 1) / 2 * shape
    return np.clip(particles, 0, shape[np.newaxis, np.newaxis, :] - 1).astype(np.uint16)


@nb.njit(nb.uint16[:,:,:](nb.uint16[:,:,:]), nogil=True)
def transitive_fast_forward(particles_map):
    """ The particle map acts as sort of map of pointers, where each entry tells us where to the flow moved a certain
        particle. We can save simulating a lot of flow steps by extending the trajectories of particles using the now
        known motion of other particles in the simulation. Iterating over this idea for a few steps we may find fixed
        points or at least fixed point sets. """

    iterations = 100
    for _ in range(iterations):
        #particles_map = particles_map[particles_map[:,:,0], particles_map[:,:,1]]
        for x in range(particles_map.shape[0]):
            for y in range(particles_map.shape[1]):
                particles_map[x, y] = particles_map[particles_map[x, y, 0], particles_map[x, y, 1]]

    return particles_map


def get_cells(particle_map: np.ndarray, size_threshold: int) -> np.ndarray:
    centroids = np.unique(particle_map.reshape((-1, 2)), axis=0)
    cell_map = np.zeros(shape=particle_map.shape[0:2], dtype=np.uint16)
    num_cell = 1
    for centroid in centroids:
        if (centroid != uint16max).all():
            cell = (particle_map == centroid[np.newaxis, np.newaxis, :]).all(axis=-1)

            cell_size = cell.sum()
            if cell_size < size_threshold:
                continue

            cell_map[cell] = num_cell
            num_cell += 1

    plt.imshow(cell_map, cmap='gist_rainbow')
    plt.show()

    return cell_map


with h5py.File(input_file, 'r') as h5file:
    flowfields = h5file['flowfields']
    labels = h5file['labels']
    label_names = [x for x in flowfields]

    for label_name in label_names:
        mask = make_binary_mask(labels[label_name])  # points in cells [HxW with 1s iff in a cell]
        flow = decode_polar_flow(flowfields[label_name])
        particle_map = simulate_flow(mask, flow, 10)  # map from pixels to cellid

        particle_map = transitive_fast_forward(particle_map)
        get_cells(particle_map, size_threshold=10)

        particle_map[mask == 0] = uint16max

        img = np.zeros_like(mask)
        for x in range(mask.shape[0]):
            for y in range(mask.shape[1]):
                if mask[x,y] == 1:
                    img[*particle_map[x, y]] = 1

        plt.imshow(img)
        plt.show()


        # mask = make_binary_mask(labels[label_name])  # points in cells [HxW with 1s iff in a cell]
        # flow = decode_polar_flow(flowfields[label_name])
        #
        # #follow_points = [[100, 100], [250, 100], [10, 10], [0, 0]]
        # follow_points = None
        #
        # niters = [1, 2, 3, 4, 5, 10, 25, 35, 50, 100, 200]
        # imgs = [flowfields[label_name]]
        # for niter in niters:
        #     print(f'At iter {niter}')
        #     cell_map = simulate_flow(mask, flow, niter)
        #     img = np.zeros_like(mask)
        #     if follow_points is None:
        #         for x in range(mask.shape[0]):
        #             for y in range(mask.shape[1]):
        #                 img[*cell_map[x,y].astype(np.uint16)] = mask[x,y]
        #     else:
        #         for fp in follow_points:
        #             print(f'After {niter} iterations point {fp} moved to {cell_map[*fp].astype(np.uint16)}')
        #             img[*cell_map[*fp].astype(np.uint16)] = 1
        #     imgs.append(img)
        #
        # num_cols = 4
        # fig, axs = plt.subplots(3, num_cols, sharex=True, sharey=True)
        # for idx, img in enumerate(imgs):
        #     axs[idx//num_cols,idx%num_cols].imshow(img)
        # plt.show()
