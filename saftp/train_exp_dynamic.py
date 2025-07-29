import os
import csv
import random
import pandas as pd
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Sampler
from tqdm import tqdm
from finger_utils import build_mlps, seed_everything, AverageMeter
import copy
from scipy.spatial.distance import euclidean

from pathlib import Path
import argparse
from collections import defaultdict
torch.set_float32_matmul_precision('high')


def parse_args():
    parser = argparse.ArgumentParser(description="Finger Trajectory Prediction")
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="Directory containing the finger trajectory datasets")
    parser.add_argument("--mode", type=str, default="mlp",
                        choices=["standard", "autoregressive", "mlp"],
                        help="Model mode: standard (default), autoregressive, or mlp")
    parser.add_argument("--window_size_min", type=int, default=3, help="Minimum window size")
    parser.add_argument("--window_size_max", type=int, default=50, help="Maximum window size")
    parser.add_argument("--val_window_size_min", type=int, default=15, help="Minimum validation window size")
    parser.add_argument("--val_window_size_max", type=int, default=15, help="Maximum validation window size")
    parser.add_argument("--teacher_forcing_ratio", type=float, default=0.5,
                        help="Ratio for teacher forcing in autoregressive training")
    parser.add_argument('--evaluate_only', action='store_true', default=False)
    return parser.parse_args()




class TransformerModel(nn.Module):
    def __init__(self, input_size, proj_dim, nhead, num_layers, ff_dim, decoder_layers, mode="standard"):
        super(TransformerModel, self).__init__()
        self.mode = mode
        self.proj_linear = nn.Linear(input_size, proj_dim)
        self.proj_nonlinear = nn.PReLU()
        self.add_layernorm = nn.LayerNorm(proj_dim)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=proj_dim, nhead=nhead, dim_feedforward=ff_dim, batch_first=True),
            num_layers=num_layers
        )
        
        # Only create decoder for non-mlp modes
        if mode != "mlp":
            self.transformer_decoder_layer = nn.TransformerDecoderLayer(d_model=proj_dim, nhead=nhead,
                                                                    dim_feedforward=ff_dim, batch_first=True)
            self.transformer_decoder = nn.TransformerDecoder(self.transformer_decoder_layer, num_layers=decoder_layers)
            self.tgt_linear = nn.Linear(2, proj_dim)
            self.tgt_nonlinear = nn.PReLU()
            self.de_fc = nn.Sequential(nn.Linear(proj_dim, int(proj_dim / 2)),
                                   nn.LeakyReLU(negative_slope=0.2),
                                   nn.Linear(int(proj_dim / 2), proj_dim),
                                   nn.LeakyReLU(negative_slope=0.2),
                                   nn.Linear(proj_dim, 2))
        
        self.en_fc = nn.Sequential(nn.Linear(proj_dim, int(proj_dim / 2)),
                               nn.LeakyReLU(negative_slope=0.1),
                               nn.Linear(int(proj_dim / 2), proj_dim),
                               nn.LeakyReLU(negative_slope=0.1),
                               nn.Linear(proj_dim, 2))

    def _generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
        return mask

    def pre_forward(self, pre_x):
        x = self.proj_nonlinear(self.proj_linear(pre_x))
        memory = self.transformer_encoder(x)
        batch_size, seq_len, hidden_dim = memory.shape
        en_output = self.en_fc(memory.reshape(batch_size*seq_len, hidden_dim))
        pre_output = en_output.reshape(batch_size, seq_len, 2)
        
        if self.mode == "mlp":
            # In MLP mode, we only return the encoder output
            return memory, pre_output
        else:
            return memory, pre_output

    def forward(self, tgt, fused_feature, use_autoregressive=False):
        if self.mode == "mlp":
            # In MLP mode, we don't use the decoder - just return what was already computed
            return None  # This will be ignored since in MLP mode we only use pre_forward output
        
        tgt = self.tgt_nonlinear(self.tgt_linear(tgt))
        
        if self.mode == "autoregressive" or use_autoregressive:
            tgt_mask = self._generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
            output = self.transformer_decoder(tgt, fused_feature, tgt_mask=tgt_mask)
        else:
            # Standard non-autoregressive mode
            output = self.transformer_decoder(tgt, fused_feature)
            
        batch_size, seq_len, hidden_dim = output.shape
        output = output.reshape(batch_size * seq_len, hidden_dim)
        output = self.de_fc(output).reshape(batch_size, seq_len, 2)
        return output



def create_dataset(finger_paths: Union[list, str]) -> List[pd.DataFrame]:
    union_finger_files = []
    if isinstance(finger_paths, str):
        finger_paths = [finger_paths]
    for finger_path in finger_paths:
        finger_files = os.listdir(finger_path)
        lonely_finger_list = [os.path.join(finger_path, finger_file) for finger_file in finger_files if
                not finger_file.startswith("._")]
        union_finger_files.extend(lonely_finger_list)
    union_finger_files = sorted(union_finger_files)
    print("len finger_files", len(union_finger_files))
    list_csv = []
    for file_destination in union_finger_files:
        try:
            data_df = pd.read_csv(file_destination, header=None, sep=",", names=["timestamp", "x", "y"],
                                  encoding="unicode_escape").iloc[::2, ] # reduce_sample_size
            if data_df.shape[0] < 5:
                print(f"find something bad in data_df with shape {data_df.shape[0]}", file_destination)
            list_csv.append(data_df)
        except Exception as e:
            print(e)
    return list_csv


def process_dataset_and_split(list_csv, window_size_range, index, stride: int = 2):
    data_dict = {length: [] for length in range(window_size_range[0], window_size_range[1] + 1)}
    target_dict = {length: [] for length in range(window_size_range[0], window_size_range[1] + 1)}
    invalid_count = 0

    for idx, csv_data in enumerate(list_csv):
        if idx not in index:
            continue
        for window_size in range(window_size_range[0], window_size_range[1] + 1):
            predict_horizon = int(window_size * 1.5)
            if csv_data.shape[0] < (window_size + predict_horizon):
                invalid_count += 1
                break
            data = csv_data[['timestamp', 'x', 'y']].values
            for i in range(0, len(data) - window_size - predict_horizon + 1, stride):
                seq = data[i:i + window_size]
                target = data[i + window_size:i + window_size + predict_horizon][:, 1:]
                data_dict[window_size].append(seq.astype(np.float32))
                target_dict[window_size].append(target.astype(np.float32))
    print("invalid_count csv file", invalid_count)
    # print(f"whole dataset {len(data_dict[15])}, {len(target_dict[15])}")
    return data_dict, target_dict


class CustomDataset(Dataset):
    def __init__(self, data_dict, target_dict, mode='train'):
        self.data_dict = data_dict
        self.target_dict = target_dict
        self.lengths = list(data_dict.keys())
        self.mode = mode
        self.length_freq = self._calculate_length_frequency()

        # Use different sampling strategies based on mode
        if mode == 'train':
            self.index_map = self._create_index_map_balanced()
        else:  # val or test - use natural distribution
            self.index_map = self._create_index_map_natural()

        self.direction_points = 1  # use point[-1] - mean(point[-3, -2]) as the rotate direction

    def _calculate_length_frequency(self):
        freq = {length: len(self.data_dict[length]) for length in self.lengths}
        total = sum(freq.values())
        return {length: count / total for length, count in freq.items()}

    def _create_index_map_balanced(self):
        """Create a balanced index map for training"""
        index_map = []
        max_freq = max(self.length_freq.values())
        for length in self.lengths:
            try:
                adjusted_count = int(max_freq / self.length_freq[length])
            except Exception as e:
                print(f"Error balancing length {length}: {e}")
                adjusted_count = 1
            for _ in range(adjusted_count):
                for idx in range(len(self.data_dict[length])):
                    index_map.append((length, idx))
        random.shuffle(index_map)
        return index_map

    def _create_index_map_natural(self):
        """Create an index map that preserves natural distribution for val/test"""
        index_map = []
        for length in self.lengths:
            for idx in range(len(self.data_dict[length])):
                index_map.append((length, idx))
        return index_map

    def normalize(self, sequence, target):
        # Same normalization logic as before
        centered_point = sequence[-1]
        prev_centered_point = np.mean(sequence[-1-self.direction_points:-1], axis=0)
        heading_vector = centered_point[1:] - prev_centered_point[1:]
        theta = np.arctan2(heading_vector[1], heading_vector[0])
        rotate_mat = np.array([[np.cos(theta), -np.sin(theta)],
                               [np.sin(theta), np.cos(theta)]])
        sequence = sequence - centered_point
        sequence[:, 1:] = np.matmul(sequence[:, 1:], rotate_mat)
        target = target - centered_point[1:]
        target = np.matmul(target, rotate_mat)
        return sequence, target, centered_point, rotate_mat

    def __getitem__(self, idx):
        length, data_idx = self.index_map[idx]
        sequence = self.data_dict[length][data_idx]
        target = self.target_dict[length][data_idx]
        sequence, target, centered_point, rotate_mat = self.normalize(sequence, target)
        return sequence, target, centered_point, rotate_mat

    def __len__(self):
        return len(self.index_map)

class SameLengthSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.batches = self._create_batches()

    def reset_batches(self):
        self.batches = self._create_batches()

    def _create_batches(self):
        # Group indices by length
        indices_by_length = {}
        for idx, (length, _) in enumerate(self.dataset.index_map):
            if length not in indices_by_length:
                indices_by_length[length] = []
            indices_by_length[length].append(idx)

        # Shuffle indices within each length group
        for length_indices in indices_by_length.values():
            np.random.shuffle(length_indices)

        # Create batches ensuring each batch contains indices of the same length
        batches = []
        for length_indices in indices_by_length.values():
            for i in range(0, len(length_indices), self.batch_size):
                if len(length_indices) <= i + self.batch_size:
                    break
                batches.append(length_indices[i:i + self.batch_size])
        return batches

    def __iter__(self):
        # Shuffle batches to ensure random order
        random.shuffle(self.batches)
        for batch in self.batches:
            for idx in batch:
                yield idx

    def __len__(self):
        return sum(len(b) for b in self.batches)

def trigger_point_distance_fn(list_csv):
    average_rmse_list = []
    for csv_data in list_csv:
        coord_data = csv_data[["x", "y"]].values
        start_point, end_point = coord_data[0], coord_data[-1]
        # compute the distance between the first and last point, divided by the number of points
        distance = euclidean(start_point, end_point) / len(coord_data)
        # distance = np.linalg.norm(end_point - start_point) / len(coord_data)
        average_rmse_list.append(distance)
    print("average rmse", np.mean(average_rmse_list), "std", np.std(average_rmse_list))
    np.save("average_two_point_distance_rmse.npy", np.array(average_rmse_list))
    exit(0)


def train_and_evaluate_models(all_file_path: Union[list, str] = None, window_size_range: tuple = (15, 15),
                              val_window_size_range: tuple = (15, 15), mode="standard",
                              teacher_forcing_ratio=0.5, trigger_point_distance: bool = False):
    batch_size = 1024
    predict_horizon, test_size, val_size = 60, 0.1, 0.1
    list_csv = create_dataset(all_file_path)

    # compute the distance averaged across all csv, with (Euclidean distance of last point and first point) / point_count
    if trigger_point_distance:
        trigger_point_distance_fn(list_csv)

    # Create train/val/test split (0.8, 0.1, 0.1)
    random_index = np.random.permutation(len(list_csv))
    test_boundary = int(len(list_csv) * test_size)
    val_boundary = int(len(list_csv) * (test_size + val_size))

    test_index = random_index[:test_boundary]
    val_index = random_index[test_boundary:val_boundary]
    train_index = random_index[val_boundary:]

    print(f"Split: Train={len(train_index)}, Val={len(val_index)}, Test={len(test_index)}")

    # Process datasets with appropriate splits
    train_data_dict, train_target_dict = process_dataset_and_split(list_csv, window_size_range, train_index)
    val_data_dict, val_target_dict = process_dataset_and_split(list_csv, val_window_size_range, val_index)
    test_data_dict, test_target_dict = process_dataset_and_split(list_csv, val_window_size_range, test_index)

    # Create datasets with appropriate modes
    train_dataset = CustomDataset(train_data_dict, train_target_dict, mode='train')
    val_dataset = CustomDataset(val_data_dict, val_target_dict, mode='val')
    test_dataset = CustomDataset(test_data_dict, test_target_dict, mode='test')
    print(f"Dataset sizes: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    # Use balanced sampler only for training
    train_sampler = SameLengthSampler(train_dataset, batch_size)

    # For val and test, use standard batching (no specialized sampler)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Create model, optimizer, etc. (same as original)
    input_size = 3
    nhead, num_encoder_layers = 2, 2
    proj_dim, ff_dim = 64, 64
    decoder_layers = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerModel(input_size, proj_dim, nhead, num_encoder_layers, ff_dim, decoder_layers, mode=mode)
    print("model parameters", sum(p.numel() for p in model.parameters()))
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    # Training parameters
    num_epochs = 200
    best_val_loss = float('inf')
    patience = 40
    early_stop_counter = 0
    if mode == "autoregressive":
        save_mode = f"autoregressive_{teacher_forcing_ratio}"
    else:
        save_mode = mode
    if not args.evaluate_only:
        for epoch in range(num_epochs):
            # Training
            model.train()
            if hasattr(train_loader.sampler, 'reset_batches'):
                train_loader.sampler.reset_batches()
            train_loss_keeper = AverageMeter()

            for inputs, targets, center_points, _ in tqdm(train_loader, desc=f"Epoch {epoch + 1} training"):
                # Training code (same as original)
                inputs, targets, center_points = inputs.to(device), targets.to(device), center_points.to(device)
                optimizer.zero_grad()
                bs, seq_len = inputs.shape[:2]
                fused_feature, pre_output = model.pre_forward(inputs)

                if mode == "mlp":
                    # In MLP mode, we only predict the input sequence length
                    outputs = pre_output  # Use only encoder outputs
                    outputs = torch.cumsum(outputs, dim=1)
                    loss = criterion(outputs, targets[:, :inputs.shape[1]])
                else:
                    # For standard and autoregressive modes
                    velocity_targets = targets[:, inputs.shape[1]:] - targets[:, inputs.shape[1] - 1:-1]

                    if mode == "autoregressive":
                        # Use teacher forcing based on ratio
                        if random.random() < teacher_forcing_ratio:
                            # Teacher forcing: use ground truth as input
                            tgt = velocity_targets
                        else:
                            # No teacher forcing: use network outputs
                            tgt = pre_output[:, -1:, :]
                            for i in range(velocity_targets.size(1)-1):
                                current_output = model(tgt, fused_feature, use_autoregressive=True)
                                tgt = torch.cat([tgt, current_output[:, -1:, :]], dim=1)
                        post_output = model(tgt, fused_feature, use_autoregressive=True)
                    else:
                        # Standard non-autoregressive behavior
                        tgt = torch.cat([pre_output[:, -1:, :], velocity_targets[:, :-1]], dim=1)
                        tgt = torch.randn_like(tgt)
                        post_output = model(tgt, fused_feature, use_autoregressive=False)

                    outputs = torch.cat([pre_output, post_output], dim=1)
                    outputs = torch.cumsum(outputs, dim=1)
                    loss = criterion(outputs, targets)

                train_loss_keeper.update(loss.item())
                loss.backward()
                optimizer.step()

            # Validation
            val_loss = evaluate(model, val_loader, criterion, device, mode)


            print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss_keeper.avg:.4f}, Val Loss: {val_loss:.4f}')

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(),
                           f'best_model_{save_mode}_ws{window_size_range[0]}_ws{window_size_range[1]}.pth')
                early_stop_counter = 0
            else:
                early_stop_counter += 1

            torch.save(model.state_dict(),
                       f'latest_model_{save_mode}_ws{window_size_range[0]}_ws{window_size_range[1]}.pth')

            # Early stopping
            if early_stop_counter >= patience:
                print(f'Early stopping after {patience} epochs without improvement. Best val loss: {best_val_loss:.4f}')
                break

        model.load_state_dict(torch.load(f'best_model_{save_mode}_ws{window_size_range[0]}_ws{window_size_range[1]}.pth'))
        test_loss = evaluate(model, test_loader, criterion, device, mode)
        print(f'Final Test Loss: {test_loss:.4f}')

    else:
        model.load_state_dict(torch.load(f'best_model_{save_mode}_ws{window_size_range[0]}_ws{window_size_range[1]}.pth'))
        metrics_by_length = defaultdict(lambda: defaultdict(int))
        overall_loss, total_samples = 0, 0
        all_per_sample_data = []

        for val_window_size_length in range(val_window_size_range[0], val_window_size_range[1] + 1):
            test_data_dict, test_target_dict = process_dataset_and_split(list_csv, (val_window_size_length, val_window_size_length), test_index)
            test_dataset = CustomDataset(test_data_dict, test_target_dict, mode='test')
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            test_loss, per_sample_data = evaluate(model, test_loader, criterion, device, mode, recursive_evaluate=True, save_per_sample=True)
            for sample in per_sample_data:
                sample['window_size'] = val_window_size_length
            all_per_sample_data.extend(per_sample_data)

            metrics_by_length[val_window_size_length]["count"] = len(test_dataset)
            metrics_by_length[val_window_size_length]["loss"] = test_loss * len(test_dataset)
            overall_loss += test_loss * len(test_dataset)
            total_samples += len(test_dataset)
            print(f'Test Loss for window size {val_window_size_length}: {test_loss:.4f}')
        df = pd.DataFrame(all_per_sample_data)
        df.to_csv(f'per_sample_metrics_{save_mode}.csv', index=False)
        print(f"Saved per-sample metrics to per_sample_metrics_{save_mode}.csv")
        avg_metrics_by_length = {}
        for length, metrics in metrics_by_length.items():
            avg_metrics_by_length[length] = np.round(np.sqrt(metrics["loss"] / metrics["count"]), 3)

        threshold_k = 10
        short_metrics = [v for k, v in avg_metrics_by_length.items() if k < threshold_k]
        long_metrics = [v for k, v in avg_metrics_by_length.items() if k >= threshold_k]

        short_rmse = np.mean(short_metrics).round(3) if short_metrics else 0
        long_rmse = np.mean(long_metrics).round(3) if long_metrics else 0
        avg_rmse = np.mean(list(avg_metrics_by_length.values())).round(3)

        print(f"Overall RMSE: {np.sqrt(overall_loss / total_samples):.3f}, total samples: {total_samples}")
        print(f"RMSE by length: {avg_metrics_by_length}")
        print(
            f"Group-wise RMSE - Short (<{threshold_k}): {short_rmse}, Long (≥{threshold_k}): {long_rmse}, Avg: {avg_rmse}")


    return model, val_dataset, test_dataset


def evaluate(model, data_loader, criterion, device, mode, recursive_evaluate=False, save_per_sample=False):
    """Evaluate model on a dataset"""
    model.eval()
    total_loss = 0
    total_samples = 0
    per_sample_data = []
    with torch.no_grad():
        for inputs, targets, center_points, _ in tqdm(data_loader, desc="Evaluating"):
            inputs, targets, center_points = inputs.to(device), targets.to(device), center_points.to(device)
            bs, seq_len = inputs.shape[:2]
            fused_feature, pre_output = model.pre_forward(inputs)

            if mode == "mlp":
                # MLP mode evaluation
                outputs = pre_output
                if recursive_evaluate: # we handle pre_output < target length case
                    if outputs.shape[1] < targets.shape[1]:
                        temp_output = outputs
                        temp_output = torch.cat([inputs[:, -temp_output.shape[1]:, 0:1], temp_output], dim=-1)
                        new_inputs = temp_output[:, -(targets.shape[1] - temp_output.shape[1]):]
                        _, new_outputs = model.pre_forward(new_inputs)
                        outputs = torch.cat([outputs, new_outputs], dim=1)
                        outputs = torch.cumsum(outputs, dim=1)

                    loss = criterion(outputs, targets)
                else:
                    outputs = torch.cumsum(outputs, dim=1)
                    loss = criterion(outputs, targets[:, :inputs.shape[1]])
            else:
                # Standard or autoregressive mode
                max_length = targets.shape[1] - inputs.shape[1]

                if mode == "autoregressive":
                    # Autoregressive generation for evaluation
                    tgt = pre_output[:, -1:, :]
                    all_outputs = [pre_output]

                    for val_idx in range(max_length):
                        post_output = model(tgt, fused_feature, use_autoregressive=True)
                        next_token = post_output[:, -1:, :]
                        tgt = torch.cat((tgt, next_token), dim=1)
                        all_outputs.append(next_token)

                    post_output = torch.cat(all_outputs[1:], dim=1)
                else:
                    # Standard non-autoregressive mode
                    tgt = torch.zeros(bs, max_length, 2, device=device)
                    tgt = torch.randn_like(tgt)
                    post_output = model(tgt, fused_feature, use_autoregressive=False)

                outputs = torch.cat([pre_output, post_output], dim=1)
                outputs = torch.cumsum(outputs, dim=1)
                loss = criterion(outputs, targets)

            if save_per_sample:
                # Calculate individual losses for each sample in the batch
                sample_losses = torch.mean((outputs - targets)**2, dim=(1, 2))
                for i in range(bs):
                    per_sample_data.append({
                        'length': seq_len,
                        'loss': sample_losses[i].item()
                    })

            total_loss += loss.item() * bs
            total_samples += bs
    if save_per_sample:
        return total_loss / total_samples, per_sample_data
    return total_loss / total_samples




if __name__ == "__main__":
    args = parse_args()
    seed_everything(42)

    data_dir = Path(args.data_dir)
    model, _, _ = train_and_evaluate_models(all_file_path=[
                                               data_dir / "finger_trajectory_straight_dec_2021",
                                               data_dir / "finger_trajectory_incline_jan_2024",
                                               data_dir / "finger_trajectory_short_jan_2024"],
                                            window_size_range=(args.window_size_min, args.window_size_max),
                                            val_window_size_range=(args.val_window_size_min, args.val_window_size_max),
                                            mode=args.mode,
                                            teacher_forcing_ratio=args.teacher_forcing_ratio)