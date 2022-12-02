# This script takes in a binary image and produces a vector representation of the connected components (cells)
# This vector representation is shown as a colored image with each color representing the direction a vector points to

import cv2, sys, os
from skimage.morphology import skeletonize
import numpy as np
from matplotlib import pyplot as plt


def build_forest(skeletonized_image: np.ndarray):
    edge_list = {}
    nodes = []
    # Embed array in zero array for easy indexing
    img = np.zeros(shape=(skeletonized_image.shape[0] + 2, skeletonized_image.shape[1] + 2), dtype=np.uint8)
    img[1:-1, 1:-1] = skeletonized_image
    for x in range(1, img.shape[0] - 1):
        for y in range(1, img.shape[1] - 1):
            if img[x, y] != 0:
                node = (x-1, y-1)
                nodes.append(node)
                edge_list[node] = []
                if img[x - 1, y - 1] != 0:
                    edge_list[node].append((x-2, y-2))
                if img[x - 1, y] != 0:
                    edge_list[node].append((x-2, y-1))
                if img[x - 1, y + 1] != 0:
                    edge_list[node].append((x-2, y))
                if img[x, y - 1] != 0:
                    edge_list[node].append((x-1, y-2))
                if img[x, y + 1] != 0:
                    edge_list[node].append((x-1, y))
                if img[x + 1, y - 1] != 0:
                    edge_list[node].append((x, y-2))
                if img[x + 1, y] != 0:
                    edge_list[node].append((x, y-1))
                if img[x + 1, y + 1] != 0:
                    edge_list[node].append((x, y))
    return nodes, edge_list


def find_trees(nodes: list, edge_list: dict):
    trees = []
    node_seen = set()
    for n in nodes:
        seen = set()
        cand = []
        if n not in node_seen:
            node_seen.add(n)
            seen.add(n)
            cand.extend(edge_list[n])
            while len(cand) > 0:
                x = cand.pop(-1)
                if x not in seen:
                    seen.add(x)
                    node_seen.add(x)
                    cand.extend(edge_list[x])
            trees.append(list(seen))
    return trees


def find_reference_points(binary_image: np.ndarray):
    reference_points = []
    skeleton = np.array(skeletonize(binary_image), np.uint8)
    nodes, edge_list = build_forest(skeleton)
    trees = find_trees(nodes, edge_list)

    for tree in trees:
        img = np.zeros_like(binary_image)
        for node in tree:
            img[node[0], node[1]] = 1

        # Contract the branches of the subgraph to its, potentially cyclic, core.
        seen = set()
        while True:
            change = False
            for node in tree:
                if node not in seen:
                    if len(edge_list[node]) == 1:
                        seen.add(node)
                        del edge_list[node]
                        change = True
            if not change:
                break

        plt.imshow(img)
        plt.show()

input_file = sys.argv[1]
assert(os.path.exists(input_file))
image = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)
image = image / 255

skeleton = np.array(skeletonize(image), np.uint8)
red_skeleton = cv2.filter2D(skeleton, ddepth=8, kernel=cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
red_skeleton = red_skeleton * skeleton

find_reference_points(image)


plt.imshow(skeleton)
plt.show()

plt.imshow(red_skeleton)
plt.show()
