import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from model.temporal_model import TemporalModel
from dataset import Label
import json
from tqdm import tqdm 
import argparse
from util import vis


class DenseSamplingDataset(Dataset):
    def __init__(self, npy_file):
        self.block_data = np.load(npy_file)[:10]
        self.block_data = torch.tensor(self.block_data, dtype=torch.float32)
        self.block_data = self.block_data.permute(1, 0, 2, 3)  

    def __len__(self):
        return self.block_data.shape[0] - 4

    def __getitem__(self, idx):
        segment = self.block_data[idx:idx + 5]
        return segment

def collate_fn(batch):
    max_height = 64
    max_width = 64

    data = []
    for segment in batch:
        resized_segment = torch.nn.functional.interpolate(segment, size=(max_height, max_width), mode='bilinear', align_corners=False)
        data.append(resized_segment)

    return torch.stack(data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Temporal Model Prediction Script")
    parser.add_argument('--checkpoint_path', type=str, 
                        default='checkpoints/temporal_checkpoint_epoch_wh.pth',
                        help='Path to the model checkpoint file.')
    parser.add_argument('--output_dir', type=str, 
                        default='predict',
                        help='Directory to save prediction results.')
    parser.add_argument('--data_dir', type=str, 
                        default='dataset',
                        help='Directory containing the .npy block data.')
    parser.add_argument('--output_file', type=str, 
                        default='demo.txt',
                        help='Name of the output text file for results.')
    parser.add_argument('--threshold', type=float, 
                        default=0.6,
                        help='Threshold for classifying significant changes (sum of score1 and score2).')

    args = parser.parse_args()

    model = TemporalModel().cuda()
    model.load_state_dict(torch.load(args.checkpoint_path))
    model.eval()

    os.makedirs(args.output_dir, exist_ok=True)

    for block_id in tqdm(Label.demo_wh.keys(), desc="Processing Blocks"):
        npy_file = os.path.join(args.data_dir, f'{block_id}.npy')
        
        if not os.path.exists(npy_file):
            print(f"File {npy_file} does not exist. Skipping block {block_id}.")
            continue
        
        dataset = DenseSamplingDataset(npy_file)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

        with torch.no_grad():
            scores_0 = []
            scores_1 = []
            scores_2 = []
            predictions = []
            for inputs in tqdm(dataloader, desc=f"Predicting for Block {block_id}"):
                inputs = inputs.cuda()

                batch_size, num_segments, channels, height, width = inputs.size()
                inputs = inputs.view(-1, channels, height, width)

                features = model.resnet(inputs)
                features = features.view(batch_size, num_segments, -1)
                features = features.permute(0, 2, 1)
                temporal_features = model.temporal_conv(features)

                pooled_features = model.max_pool(temporal_features)
                pooled_features = pooled_features.view(pooled_features.size(0), -1) 

                outputs = model.fc(pooled_features)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                probabilities_class_0 =probabilities[:,0]
                probabilities_class_1 =probabilities[:,1]
                probabilities_class_2 =probabilities[:,2]
                scores_0.extend(probabilities_class_0.cpu().tolist())
                scores_1.extend(probabilities_class_1.cpu().tolist())
                scores_2.extend(probabilities_class_2.cpu().tolist())
                predicted_classes = torch.argmax(probabilities, dim=1).cpu().tolist()
                predictions.extend(predicted_classes)

        significant_changes = []
        results = {}  
        
        output_file_path = os.path.join(args.output_dir, args.output_file)
        with open(output_file_path, "a") as file:
            file.write(f"\n\n—— Block {block_id} Results ——\n")
            classification = 0
            for idx, (score0, score1, score2, pred_class) in enumerate(zip(scores_0, scores_1, scores_2, predictions)):
                center_idx = idx + 2
                output_line = f"Slice {center_idx}: Score0 = {score0:.4f}, Score1 = {score1:.4f}, Score2 = {score2:.4f}, Predicted Class = {pred_class}\n"
                print(output_line.strip())
                file.write(output_line)
                if score1 + score2 >= args.threshold:
                    significant_changes.append(center_idx)
                    if score1 >= score2:
                        classification = 1
                    else:
                        classification = 2

            results[block_id] = significant_changes
            summary = f"\nBlock {block_id} Significant Changes: {significant_changes}, class = {classification}\n{'-'*40}"
            file.write(summary + "\n")
            print(summary)
        vis.plot_block_probabilities(block_id)