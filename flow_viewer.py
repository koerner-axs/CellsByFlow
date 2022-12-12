import numpy as np
import cv2
from cellpose import plot
import tifffile
from pathlib import Path
from matplotlib import pyplot as plt


data_dir = Path('./bamicrostructure13/cellpose/')
flow_files = list(data_dir.glob('*_flows.tif'))



#from cellpose import io
#print(len(io.get_image_files(str(data_dir), mask_filter='_mask')))
#exit(0)



for file in flow_files:
    print('Showing flow from file \"' + file.stem + '\"')
    tensor = tifffile.imread(file)
    mu_hsv = plot.dx_to_circ(tensor[2:4])
    plt.imshow(mu_hsv)
    plt.show()
