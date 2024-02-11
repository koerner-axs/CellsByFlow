import cv2
import numpy as np
from . import measure


# MORPHOLOGICAL_OPEN_STRUCTURE_ELEMENT_TYPE = None
MORPHOLOGICAL_OPEN_STRUCTURE_ELEMENT_TYPE = cv2.MORPH_CROSS
# MORPHOLOGICAL_OPEN_STRUCTURE_ELEMENT_TYPE = cv2.MORPH_RECT

MORPHOLOGICAL_OPEN_NUM_ITERATIONS = 3  # 1


def find_contours(image):
    _, image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)

    if MORPHOLOGICAL_OPEN_STRUCTURE_ELEMENT_TYPE is not None:
        image = cv2.morphologyEx(image, cv2.MORPH_OPEN,
                                 cv2.getStructuringElement(MORPHOLOGICAL_OPEN_STRUCTURE_ELEMENT_TYPE, (3, 3)),
                                 iterations=MORPHOLOGICAL_OPEN_NUM_ITERATIONS)
    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    return contours


def _draw_bb(image, rect, color):
    round_pair = lambda x: (round(x[0]), round(x[1]))
    rect = np.array(list(map(round_pair, rect)), dtype=np.int32)
    cv2.line(image, tuple(rect[0]), tuple(rect[1]), color=color, thickness=1, lineType=cv2.LINE_8)
    cv2.line(image, tuple(rect[1]), tuple(rect[2]), color=color, thickness=1, lineType=cv2.LINE_8)
    cv2.line(image, tuple(rect[2]), tuple(rect[3]), color=color, thickness=1, lineType=cv2.LINE_8)
    cv2.line(image, tuple(rect[3]), tuple(rect[0]), color=color, thickness=1, lineType=cv2.LINE_8)
    return image


def _draw_normalized_cell(points: np.ndarray, image_size: int):
    round_pair = lambda x: (round(x[0]), round(x[1]))
    poly = np.array(list(map(round_pair, points * image_size)), dtype=np.int32)
    image = cv2.fillPoly(np.zeros(shape=(image_size, image_size, 1), dtype=np.uint8), poly[np.newaxis, :, :], color=255)
    return image


def draw_cells_to_image(image: np.ndarray, normalized_contours: list, color: tuple = (255, 0, 0)):
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    mask = np.zeros_like(image)

    round_pair = lambda x: (round(x[0]), round(x[1]))
    for contour in normalized_contours:
        poly = np.array(list(map(round_pair, contour * [mask.shape[1], mask.shape[0]])), dtype=np.int32)
        mask = cv2.fillPoly(mask, poly[np.newaxis, :, :], color=list(color)[::-1])
        mask = cv2.drawContours(mask, poly[np.newaxis, :, :], 0, color=0, thickness=2)

    cv2.addWeighted(image, 0.5, mask, 0.5, 0.0, dst=image)

    return image


def find_cells(prediction: np.ndarray, cell_erosion: int):
    image = (prediction * 255.0).astype(np.uint8)
    contours = find_contours(image)

    data = []
    for cell_index, contour in enumerate(contours):
        # Find edge pixels
        edge_pixels = []
        limit1, limit2 = prediction.shape[0:2]
        for idx in range(contour.shape[0]):
            point = contour[idx, 0, :]
            if point[0] <= 0 or point[0] >= limit1 - 1 or point[1] <= 0 or point[1] >= limit2 - 1:
                edge_pixels.append(idx)

        # Create binary mask showing only the current cell
        mask = np.zeros_like(image, dtype=np.uint8)
        mask = cv2.fillPoly(mask, contour[np.newaxis, :, :], 255)

        # Re-dilate cell, compensate for necessary gaps between detected cells
        mask_dilated = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=cell_erosion)
        contour, _ = cv2.findContours(mask_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = contour[0][:, 0, :]

        # Update mask. The dilation operation could have created holes in the mask. Fill them back in by drawing the
        # contour onto the mask again.
        mask_dilated = cv2.fillPoly(mask_dilated, contour[np.newaxis, :, :], 255)

        # Calculate parameters for the cell
        parameters = measure.process_cell(contour, mask_dilated, prediction,
                                          input_size=prediction.shape, resolution=-1.0, edge_pixels=edge_pixels,
                                          applied_cell_erosion=cell_erosion)

        # Calculate normalized and square fit contours for rendering purposes
        height, width = image.shape[0:2]
        contour_normalized = contour / np.array([width, height])
        contour_square_fit = measure.fit_in_square(contour, border=0.1)

        data.append((parameters, contour_normalized, contour_square_fit))

    return data
