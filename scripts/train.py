import h5py
import tqdm
import torch
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt

from scripts.network import UNetlikeNetwork

dataset_file = Path('dataset/vectorized_bams13.h5')
model_file = Path('models/model.pt')


class AugmentedDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, labels_mask, labels_flowfield):
        self.inputs = inputs
        self.labels_mask = labels_mask
        self.labels_flowfield = labels_flowfield
        self.translation_limits = (-inputs.shape[2] // 4, inputs.shape[2] // 4)

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        input, label_mask, label_flowfield = self.inputs[idx], self.labels_mask[idx], self.labels_flowfield[idx]
        # Perform random zoom, rotation, translation, and flip on the numpy tensors
        #angle = np.random.uniform(-180, 180)
        #scale = np.random.uniform(0.5, 1.5)
        #translation = np.random.uniform(self.translation_limits, size=2).tolist()
        flip = np.random.randint(0, 2, size=2).tolist()

        if flip[0] == 1:
            input = torch.flipud(input)
            label_mask = torch.flipud(label_mask)
            label_flowfield[0] = torch.flipud(label_flowfield[0])
            label_flowfield[1] = torch.flipud(label_flowfield[1])
        if flip[1] == 1:
            input = torch.fliplr(input)
            label_mask = torch.fliplr(label_mask)
            label_flowfield[0] = torch.fliplr(label_flowfield[0])
            label_flowfield[1] = torch.fliplr(label_flowfield[1])

        return input, label_mask, label_flowfield


def train_model(model, device, epochs, batch_size, learning_rate, dataset):
    inputs, labels_mask, labels_flowfield = dataset

    inputs = inputs[:, np.newaxis, :, :].astype(np.float32)
    labels_mask = labels_mask[:, np.newaxis, :, :].astype(np.float32)
    labels_flowfield = np.moveaxis(labels_flowfield, 3, 1).astype(np.float32)

    inputs = torch.from_numpy(inputs).float().to(device)
    labels_mask = torch.from_numpy(labels_mask).float().to(device)
    labels_flowfield = torch.from_numpy(labels_flowfield).float().to(device)

    params = {'batch_size': batch_size, 'shuffle': True}#, 'num_workers': 6}

    train_dataset = AugmentedDataset(inputs, labels_mask, labels_flowfield)
    #train_dataset = torch.utils.data.TensorDataset(inputs, labels_mask, labels_flowfield)
    train_loader = torch.utils.data.DataLoader(train_dataset, **params)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    grad_scaler = torch.cuda.amp.GradScaler()
    loss_fn_mask = torch.nn.BCELoss()
    loss_fn_flowfield = torch.nn.MSELoss()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm.tqdm(total=len(train_loader), desc=f'Epoch {epoch}/{epochs}', unit='batch', leave=True) as pbar:
            for batch_idx, (input, target_mask, target_flowfield) in enumerate(train_loader):

                #plt.imshow(input[0, 0, :, :].cpu().numpy())
                #plt.show()
                #plt.imshow(target_mask[0, 0, :, :].cpu().numpy())
                #plt.show()
                #plt.imshow(target_flowfield[0, 0, :, :].cpu().numpy())
                #plt.show()
                #plt.imshow(target_flowfield[0, 1, :, :].cpu().numpy())
                #plt.show()

                pred_mask, pred_flowfield = model(input)
                loss = loss_fn_mask(pred_mask, target_mask)
                loss += loss_fn_flowfield(pred_flowfield, target_flowfield)
                optimizer.zero_grad()
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()
                epoch_loss += loss.item()
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                pbar.update()


if __name__ == '__main__':
    with h5py.File(dataset_file, 'r') as h5file:
        inputs = h5file['inputs']
        labels = h5file['labels_binary']
        flowfields = h5file['flowfields']
        input_file_names = [x for x in inputs]

        print(f'There are {len(input_file_names)} input files in the dataset.')
        print(f'The flow fields were generated at {flowfields.attrs["generated_at"]}.')

        inputs = np.array([inputs[x][:] for x in input_file_names])
        labels = np.array([labels[x][:] for x in input_file_names])
        flowfields = np.array([flowfields[x + '_vector'][:] for x in input_file_names])

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = UNetlikeNetwork()

        model.to(device)
        train_model(model, device, epochs=1000, batch_size=64, learning_rate=1e-4, dataset=(inputs, labels, flowfields))
        model.cpu()

        torch.save(model.state_dict(), model_file)

        # for file_name in input_file_names:
        #     mask = make_binary_mask(labels[label_name])  # points in cells [HxW with 1s iff in a cell]
        #     flow = decode_polar_flow(flowfields[label_name])
        #     particle_map = simulate_flow(mask, flow, 10)  # map from pixels to cellid
        #
        #     particle_map = transitive_fast_forward(particle_map)
        #     get_cells(particle_map, size_threshold=10)
        #
        #     particle_map[mask == 0] = uint16max
        #
        #     img = np.zeros_like(mask)
        #     for x in range(mask.shape[0]):
        #         for y in range(mask.shape[1]):
        #             if mask[x,y] == 1:
        #                 img[*particle_map[x, y]] = 1
        #
        #     plt.imshow(img)
        #     plt.show()
