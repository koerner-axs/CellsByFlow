import numpy as np
import cv2

ALLOW_NONSENSE_CPP_AND_CPPE = True


def _dist_euclidean(p1: np.ndarray, p2: np.ndarray) -> float:
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)


def _ellipse_circumference_ramanujan(a, b) -> float:
    return (np.pi / 2.0) * ((a+b) + (3*((a-b)**2)) / (10*(a+b) + np.sqrt(a*a + 14*a*b + b*b)))


def _find_min_area_bb(points: np.ndarray) -> tuple:
    # Find minimum area bounding box for set of points by rotating in increments of
    # half a degree from 0.0 to 90.0 degrees.
    # Since only the contour is given this should not be too expensive.

    best_rect = None
    for current_angle in np.linspace(0.0, np.pi/2, 180, endpoint=False):
        rotation_matrix = np.array([[np.cos(current_angle), -np.sin(current_angle)],
                                    [np.sin(current_angle), np.cos(current_angle)]])
        rotated_points = np.einsum('AB,NB->NA', rotation_matrix, points)
        minX, minY = rotated_points[:,0].min(), rotated_points[:,1].min()
        maxX, maxY = rotated_points[:,0].max(), rotated_points[:,1].max()
        area = (maxX-minX+1)*(maxY-minY+1) # adjust by one pixel to account for points being at the center of pixels
        if best_rect is None or best_rect[0] > area:
            best_rect = (area, current_angle, minX, minY, maxX, maxY)

    area, angle, minX, minY, maxX, maxY = best_rect
    long_axis = max(maxX-minX, maxY-minY) + 1 # adjust by one pixel to account for points being at the center of pixels
    short_axis = min(maxX-minX, maxY-minY) + 1 # same adjustment
    corners = np.array([[minX, minY], [minX, maxY], [maxX, maxY], [maxX, minY]])
    rotation_matrix = np.array([[np.cos(-angle), -np.sin(-angle)],
                                [np.sin(-angle), np.cos(-angle)]])
    corners = np.einsum('AB,NB->NA', rotation_matrix, corners)

    return angle, corners, long_axis, short_axis


def _find_min_width_bb(points: np.ndarray) -> tuple:
    # Find minimum width bounding box for set of points by rotating in increments of
    # half a degree from 0.0 to 90.0 degrees.
    # Since only the contour is given this should not be too expensive.

    best_rect = None
    for current_angle in np.linspace(0.0, np.pi/2, 180, endpoint=False):
        rotation_matrix = np.array([[np.cos(current_angle), -np.sin(current_angle)],
                                    [np.sin(current_angle), np.cos(current_angle)]])
        rotated_points = np.einsum('AB,NB->NA', rotation_matrix, points)
        minX, minY = rotated_points[:,0].min(), rotated_points[:,1].min()
        maxX, maxY = rotated_points[:,0].max(), rotated_points[:,1].max()
        area = (maxX-minX+1)*(maxY-minY+1) # adjust by one pixel to account for points being at the center of pixels
        small_axis = min(maxX-minX, maxY-minY) + 1 # adjust by one pixel to account for points being at the center of pixels
        if best_rect is None or best_rect[1] > small_axis or (best_rect[1] == small_axis and best_rect[0] > area):
            best_rect = (area, small_axis, current_angle, minX, minY, maxX, maxY)

    area, _, angle, minX, minY, maxX, maxY = best_rect
    long_axis = max(maxX-minX, maxY-minY) + 1 # adjust by one pixel to account for points being at the center of pixels
    short_axis = min(maxX-minX, maxY-minY) + 1 # same adjustment
    corners = np.array([[minX, minY], [minX, maxY], [maxX, maxY], [maxX, minY]])
    rotation_matrix = np.array([[np.cos(-angle), -np.sin(-angle)],
                                [np.sin(-angle), np.cos(-angle)]])
    corners = np.einsum('AB,NB->NA', rotation_matrix, corners)

    return angle, corners, long_axis, short_axis


def fit_in_square(points: np.ndarray, border: float) -> np.ndarray:
    # Find a square bounding box for set of points by rotating in increments of
    # half a degree from 0.0 to 90.0 degrees.
    # Since only the contour is given this should not be too expensive.
    # Return points projected into the square.
    # Output will be array with shape of points, but with values in [0.0;1.0]
    # to indicate point location in the computed square.
    best_rect = None
    for current_angle in np.linspace(0.0, np.pi/2, 180, endpoint=False):
        rotation_matrix = np.array([[np.cos(current_angle), -np.sin(current_angle)],
                                    [np.sin(current_angle), np.cos(current_angle)]])
        rotated_points = np.einsum('AB,NB->NA', rotation_matrix, points)
        minX, minY = rotated_points[:,0].min(), rotated_points[:,1].min()
        maxX, maxY = rotated_points[:,0].max(), rotated_points[:,1].max()
        area = (maxX-minX+1)*(maxY-minY+1) # adjust by one pixel to account for points being at the center of pixels
        aspect_ratio = max(maxX-minX+1, maxY-minY+1) / min(maxX-minX+1, maxY-minY+1)
        if best_rect is None or best_rect[1] > aspect_ratio or (best_rect[1] == aspect_ratio and best_rect[0] > area):
            best_rect = (area, aspect_ratio, current_angle, minX, minY, maxX, maxY)

    area, _, angle, minX, minY, maxX, maxY = best_rect
    
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                [np.sin(angle), np.cos(angle)]])
    projected_points = np.einsum('AB,NB->NA', rotation_matrix, points)

    if (maxX-minX) >= (maxY-minY):
        scaling_factor = (1.0 - 2*border) / (maxX-minX+1)
    else:
        scaling_factor = (1.0 - 2*border) / (maxY-minY+1)

    projected_points -= 0.5 * projected_points.max(axis=0) + 0.5 * projected_points.min(axis=0)
    projected_points = projected_points * scaling_factor
    projected_points[:,0] += 0.5
    projected_points[:,1] += 0.5

    return projected_points


def process_cell(contour: np.ndarray, mask: np.ndarray, network_prediction: np.ndarray, input_size: tuple,
                 resolution: float, edge_pixels: list, applied_cell_erosion: int) -> dict:
    """ Process one cell. Return a dictionary, which maps the name of a metric to its value:
            {'perimeter': 98.3, 'area': 234.0, 'is_edge_cell': True, etc.}
        Parameters:
            contour: [N,2] array of pixels that make up the border of the detected cell.
            mask: [H,W] boolean array indicating the location of the cell. Contour included.
            network_prediction: [H,W] float array. Prediction of the segmentation network. Values in [0.0, 1.0].
                                Higher values mean higher likelihood of a cell pixel.
            input_size: tuple. (H,W).
            resolution: float. Resolution of the input image in nm/px.
            edge_pixels: [N] array of indices into array 'contour'. These pixels lie on the edge of the input image.
            applied_cell_erosion: int. Number of pixels the detected cell is to small by in every direction.
                                  This is by design to allow for the separation of the cells. """

    # Calculate area
    area = float(np.count_nonzero(mask))

    # Calculate perimeter
    perimeter = cv2.arcLength(contour[:, np.newaxis, :], True)

    # Compute the convex hull to speed up computation
    convex_hull = cv2.convexHull(contour, clockwise=False, returnPoints=True)
    convex_hull = convex_hull[:, 0, :]

    # Calculate minimum width bounding box
    angle, rect, long_axis, short_axis = _find_min_width_bb(contour)
    aspect_ratio = long_axis / short_axis

    # Calculate C_PP
    c_pp = 4 * np.pi * area / (perimeter * perimeter)
    if c_pp > 1.0 and not const.ALLOW_NONSENSE_CPP_AND_CPPE:
        #print('C_PP was', c_pp)
        #print('Adjusted to 1.0 as values greater 1.0 do not make sense and constitute an error of precision.')
        c_pp = 1.0

    # Calculate C_PPE (essentially C_PP adjusted for aspect ratio, otherwise ellipses would score poorly)
    # Calculate axis length for an ellipse of same area and aspect ratio
    ellipsis_long_axis = 2 * np.sqrt(area * long_axis / (np.pi * short_axis))
    ellipsis_short_axis = 2 * np.sqrt(area * short_axis / (np.pi * long_axis))
    c_ppe = (_ellipse_circumference_ramanujan(ellipsis_long_axis, ellipsis_short_axis) / perimeter)**2
    if c_ppe > 1.0 and not const.ALLOW_NONSENSE_CPP_AND_CPPE:
        #print('C_PPE was', c_ppe)
        #print('Adjusted to 1.0 as values greater 1.0 do not make sense and constitute an error of precision.')
        c_ppe = 1.0

    # Calculate C_PCH
    perimeter_convex_hull = cv2.arcLength(convex_hull[:, np.newaxis, :], True)
    #c_pch = np.clip(perimeter_convex_hull / perimeter, 0.0, 1.0)
    c_pch = perimeter_convex_hull / perimeter

    # Calculate C_ACH
    convex_hull_binary_img = np.zeros_like(mask, dtype=np.uint8)
    convex_hull_binary_img = cv2.fillPoly(convex_hull_binary_img, convex_hull[np.newaxis, :, :], 255)
    area_convex_hull = float(np.count_nonzero(convex_hull_binary_img))
    #c_ach = np.clip(area / area_convex_hull, 0.0, 1.0)
    c_ach = area / area_convex_hull

    # Decide if this cell touches the edge enough to need to be considered an edge cell
    is_edge_cell = len(edge_pixels) > 0

    return {'Area': area, 'Perimeter': perimeter, 'Convex Hull Perimeter': perimeter_convex_hull,
            'Convex Hull Area': area_convex_hull, 'Aspect ratio': aspect_ratio,
            'Long axis': long_axis, 'Short axis': short_axis,
            'Compactness Polsby-Popper': c_pp,
            'Compactness Elliptic Polsby-Popper': c_ppe,
            'Compactness Perimeter Convex Hull': c_pch,
            'Compactness Area Convex Hull': c_ach,
            'Is edge cell': is_edge_cell}


def get_all_metric_names():
    return ['Area', 'Perimeter', 'Convex Hull Perimeter', 'Convex Hull Area', 'Aspect ratio', 'Long axis', 'Short axis',
            'Compactness Polsby-Popper', 'Compactness Elliptic Polsby-Popper',
            'Compactness Perimeter Convex Hull', 'Compactness Area Convex Hull',
            'Is edge cell']
