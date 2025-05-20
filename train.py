import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from model.temporal_model import TemporalModel
import random
import wandb
from dataset import Label
import argparse
from tqdm import tqdm


class SegmentDataset(Dataset):
    def __init__(self, block_list, no_change, train1, train2, transform=None, target_length=5):
        self.block_list = block_list
        self.no_change = no_change
        self.train1 = train1
        self.train2 = train2
        self.transform = transform
        self.target_length = target_length

    def __len__(self):
        return len(self.block_list)

    def __getitem__(self, idx):
        # 随机选择变化或无变化
        class_change = 0
        if random.random() < 0.5:
            block_id = random.choice(list(self.no_change.keys()))
            label = self.no_change.get(block_id, [])
        else:
            if random.random() < 0.5:
                block_id = random.choice(list(self.train1.keys()))
                label = self.train1.get(block_id, [])
                class_change = 1
            else:
                block_id = random.choice(list(self.train2.keys()))
                label = self.train2.get(block_id, [])
                class_change = 2

        file_path = f'dataset/{block_id}/data.npy'
        block_data = np.load(file_path)[:10]
        block_data = block_data[:, :32]
        block_data = torch.tensor(block_data, dtype=torch.float32)
        block_data = block_data.permute(1, 0, 2, 3)

        segments = []
        labels = []

        for i in range(1, block_data.shape[0] - 1):
            # segment = block_data[i - 2:i + 3]
            segment = block_data[i - 1:i + 2]
            segments.append(segment)

            if any(start <= i <= end for start, end in label):
                labels.append(class_change)  # 变化
            else:
                labels.append(0)  # 无变化

        return torch.stack(segments), torch.tensor(labels)


def collate_fn(batch):
    max_height = 64
    max_width = 64

    data = []
    labels = []
    for segments, label in batch:
        for segment in segments:
            resized_segment = torch.nn.functional.interpolate(segment, size=(max_height, max_width), mode='bilinear',
                                                              align_corners=False)
            data.append(resized_segment)
        labels.extend(label)

    return torch.stack(data), torch.tensor(labels)


def main():
    parser = argparse.ArgumentParser(description='Train a temporal model for change detection.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training.')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs.')
    parser.add_argument('--block_count', type=int, default=400, help='Number of blocks to use in the dataset.')
    parser.add_argument('--save_interval', type=int, default=20, help='Save model checkpoint every N epochs.')
    args = parser.parse_args()

    # Initialize wandb
    wandb.init(project='temporal_model_training', config=args)

    block_list = list(range(args.block_count))
    no_change = Label.no_change
    train1 = Label.train_1
    train2 = Label.train_2

    dataset = SegmentDataset(block_list, no_change, train1, train2)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    model = TemporalModel().cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    num_epochs = args.num_epochs

    for epoch in tqdm(range(num_epochs), desc="Training Epochs"):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        for inputs, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            inputs = inputs.cuda()
            labels = labels.cuda()

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

            wandb.log({"batch_loss": loss.item()})

        avg_loss = running_loss / len(dataloader)
        epoch_accuracy = 100 * correct_predictions / total_predictions

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%')

        # Logging epoch metrics to wandb
        wandb.log({"epoch_loss": avg_loss, "epoch_accuracy": epoch_accuracy, "epoch": epoch + 1})

        if (epoch + 1) % args.save_interval == 0:
            os.makedirs('checkpoints', exist_ok=True)
            checkpoint_path = f'checkpoints/temporal_checkpoint_epoch{epoch + 1}.pth'
            torch.save(model.state_dict(), checkpoint_path)
            wandb.save(checkpoint_path)

    # Finish the wandb run
    wandb.finish()


if __name__ == '__main__':
    main()