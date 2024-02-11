import h5py
import torch
import cv2
import numpy as np
from scripts.network import UNetlikeNetwork
from pathlib import Path
from matplotlib import pyplot as plt


model_file = Path('models/model.pt')
dataset_file = Path('dataset/vectorized_bams13.h5')


def load_model():
    model = UNetlikeNetwork()
    model.load_state_dict(torch.load(model_file))
    model.eval()
    return model


def predict(model, input):
    with torch.no_grad():
        pred_mask, pred_flowfield = model(input)
    return pred_mask, pred_flowfield


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model()
    model.to(device)

    with h5py.File(dataset_file, 'r') as h5file:
        input_names = [x for x in h5file['inputs']]
        inputs = np.array([h5file['inputs'][x][:] for x in input_names])

        # Perform CLAHE preprocessing
        #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        #for i in range(inputs.shape[0]):
        #    inputs[i, :, :] = clahe.apply(inputs[i, :, :])

        inputs = torch.from_numpy(inputs).to(device).float()
        inputs = inputs.unsqueeze(1)
        pred_mask, pred_flowfield = predict(model, inputs)
        pred_mask = pred_mask.cpu().numpy()
        pred_flowfield = pred_flowfield.cpu().numpy()

    for i, name in enumerate(input_names):
        pred_mask_i = pred_mask[i, 0, :, :]
        cv2.imwrite(f'predictions/{name}_mask.png', (pred_mask_i > 0.5).astype(np.uint8) * 255)
        pred_flowfield_dx = pred_flowfield[i, 0, :, :]
        pred_flowfield_dy = pred_flowfield[i, 1, :, :]
        pred_flow_polar = np.arctan2(pred_flowfield_dx, pred_flowfield_dy)
        pred_flow_polar = (np.nan_to_num(pred_flow_polar) + np.pi) / (2 * np.pi)
        #pred_flow_polar = np.stack([pred_flow_polar, np.ones_like(pred_flow_polar), np.ones_like(pred_flow_polar)],
        #                           axis=-1)
        #pred_flow_polar = cv2.cvtColor(pred_flow_polar, cv2.COLOR_HSV2RGB)
        #polar_flow_bw = np.stack([pred_flow_polar] * 3, axis=-1)
        #cv2.imwrite(f'predictions/{name}_flow.png', polar_flow_bw.astype(np.uint8) * 255)

        plt.imsave(f'predictions/{name}_flow.png', pred_flow_polar, cmap='gist_rainbow')

        fig, ax = plt.subplots(2, 2)
        ax[0,0].imshow(pred_flowfield_dx)
        ax[0,1].imshow(pred_flowfield_dy)
        ax[1,0].imshow(pred_flow_polar)
        ax[1,1].imshow(pred_mask_i)
        plt.show()
