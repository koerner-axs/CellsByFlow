import numpy as np
import cv2
import xml
from pathlib import Path


dst_dir = Path('./bamicrostructure13/cellpose/')
inputs_dir = Path('./bamicrostructure13/raw/JPEGImages/')
labels_dir = Path('./bamicrostructure13/raw/SegmentationObject/')


def convert_label_to_canonical_numbering(img: np.ndarray):
    height, width = img.shape[0:2]
    numbering = np.zeros(shape=(height, width), dtype=np.uint16)

    colors = np.unique(img.reshape((-1, 3)), axis=0)
    for index, color in enumerate(colors):
        if (color == 0).all():
            continue  # Background color (0,0,0) does not need to be checked.
        numbering[(img == color).all(axis=-1)] = index

    return numbering


def load_labels(directory: Path):
    xml_files = list(directory.glob('*.xml'))
    labels = dict()

    for file in xml_files:
        root_node = xml.dom.minidom.parse(str(file)).documentElement

        for image_element in root_node.getElementsByTagName('image'):
            name = image_element.getAttribute('name').replace('.png', '')
            height, width = int(image_element.getAttribute('height')), int(image_element.getAttribute('width'))

            polygon_elements = image_element.getElementsByTagName('polygon')
            image = np.zeros((height, width), dtype=np.uint16)

            polygons = []

            for polygon_element in polygon_elements:
                text = polygon_element.getAttribute('points')
                points = [x.split(',') for x in text.split(';')]
                points = np.array([[int(np.round(float(x[0]))), int(np.round(float(x[1])))] for x in points])
                #points = np.array([[float(x[0]), float(x[1])] for x in points])
                polygons.append(points)

            for idx in range(len(polygon_elements)):
                image = cv2.fillPoly(image, polygons[idx][np.newaxis, :, :], idx, cv2.LINE_8)

            labels[name] = image

    return labels


labels = load_labels(Path('./bamicrostructure13/raw/cvat export/'))

for input_file in inputs_dir.glob('*.png'):
    print('Handling file pair \"' + input_file.stem + '\"')
    out_filename = input_file.stem.replace('_cropped', '')

    img = cv2.imread(str(input_file), cv2.IMREAD_GRAYSCALE)
    cv2.imwrite(str(dst_dir / (out_filename + '.tif')), img)

    #img = cv2.imread(str(labels_dir / (input_file.stem + '.png')), cv2.IMREAD_COLOR)
    #img = convert_label_to_canonical_numbering(img)
    img = labels[input_file.stem]
    cv2.imwrite(str(dst_dir / (out_filename + '_mask.tif')), img)#, cv2.CV_16UC1)
