# Automatic Microstructure Evaluation

## Approach
To perform segmentation of the input images into cells a U-Net inspired fully convolutional network is applied to predict flow fields.
For every pixel of the input a simulation of the dynamics in the flow field is performed. Pixels that trend toward the same point are assigned to the same cells.

## Dataset
The dataset is unfortunately non-public for now. Any derivatives such as the models and its predictions also may not be published. Once this method is further refined, I expect this to change.

## See also
Project Cellpose which proposed this or at least a similar method to segment biological cells.
