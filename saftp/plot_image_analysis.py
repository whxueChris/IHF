# latency_comparison.py
import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from tqdm import tqdm
from train_exp_dynamic import TransformerModel
from train_pure_decoder_dynamic import TransformerModel as TransformerModelPureDecoder
from finger_utils import seed_everything
import pandas as pd


plt.rcParams['font.family'] = 'Arial' # or 'sans-serif', 'Arial' depending on desired look
plt.rcParams['font.size'] = 15*9/11  # Adjust as needed

colors = ['#89CFF0', '#bdbdbd', '#75A9CB', "#DFA451", "#DAD1C0", "#BA7C4D", "#DFA451", "#DAD1C0", "#BA7C4D"]

def parse_args():
    parser = argparse.ArgumentParser(description="Latency Comparison for Model Variants")
    parser.add_argument("--window_size", type=int, default=10, help="Input sequence length")
    parser.add_argument("--min_output_length", type=int, default=10, help="Min output sequence length")
    parser.add_argument("--max_output_length", type=int, default=60, help="Max output sequence length")
    parser.add_argument("--step_size", type=int, default=5, help="Step size for output length")
    parser.add_argument("--num_runs", type=int, default=100, help="Number of runs for each test")
    parser.add_argument("--save_path", type=str, default="latency_comparison_rtx4090_lowregion.png", help="Path to save plot")
    return parser.parse_args()


def generate_tgt_dict(args, device):
    """Generate target dictionary."""
    tgt_dict = {}
    for i in range(args.min_output_length - args.window_size, args.max_output_length + 1 - args.window_size, args.step_size):
        tgt_dict[i] = torch.randn(1, i+1, 2, device=device)
    return tgt_dict


def load_model(model_type, device):
    """Load model of specified type."""
    input_size = 3
    proj_dim, ff_dim = 64, 64
    nhead, num_encoder_layers = 2, 2
    decoder_layers = 2
    if model_type != "TDec":
        model = TransformerModel(input_size, proj_dim, nhead, num_encoder_layers,
                                ff_dim, decoder_layers, mode=model_type)
    else:
        model = TransformerModelPureDecoder(input_size, proj_dim, nhead, num_encoder_layers,
                                            ff_dim, decoder_layers, mode=model_type)

    model_path = f'best_model_{model_type}_ws15_ws15.pth'
    if model_type == "autoregressive":
        model_path = f'best_model_autoregressive_1.0_ws15_ws15.pth'

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
    else:
        print(f"No pre-trained model found at {model_path}")

    model.to(device)
    model.eval()
    return model

def measure_mlp_inference(model, inputs, output_length, device):
    """Measure MLP inference with recursive evaluation if needed."""
    with torch.no_grad():
        input_length = inputs.shape[1]
        fused_feature, pre_output = model.pre_forward(inputs)

        if output_length > input_length:
            outputs = pre_output
            current_length = outputs.shape[1]

            while current_length < output_length:
                # Create new input from previous outputs
                temp_output = outputs[:, -input_length:] if outputs.shape[1] >= input_length else outputs
                temp_input = torch.cat([inputs[:, -temp_output.shape[1]:, 0:1], temp_output], dim=-1)
                _, new_outputs = model.pre_forward(temp_input)
                outputs = torch.cat([outputs, new_outputs], dim=1)

                # Check progress
                if current_length == outputs.shape[1]:
                    break  # No progress made
                current_length = outputs.shape[1]

            outputs = outputs[:, :output_length]  # Trim to exact length
        else:
            outputs = pre_output[:, :output_length]

    return torch.cumsum(outputs, dim=1)

def measure_autoregressive_inference(model, inputs, output_length, device):
    """Measure autoregressive inference time."""
    with torch.no_grad():
        fused_feature, pre_output = model.pre_forward(inputs)

        # Start with the last prediction from encoder
        tgt = pre_output[:, -1:, :]
        all_outputs = [pre_output]

        # Generate autoregressive outputs
        remaining_length = max(0, output_length - pre_output.shape[1])
        for _ in range(remaining_length):
            post_output = model(tgt, fused_feature, use_autoregressive=True)
            next_token = post_output[:, -1:, :]
            tgt = torch.cat((tgt, next_token), dim=1)
            all_outputs.append(next_token)

        if remaining_length > 0:
            post_output = torch.cat(all_outputs[1:], dim=1)
            outputs = torch.cat([pre_output, post_output], dim=1)
        else:
            outputs = pre_output

    return torch.cumsum(outputs[:, :output_length], dim=1)

def measure_pure_decoder_inference(model, inputs, output_length, device):
    """Measure pure decoder inference time."""
    with torch.no_grad():
        tgt = torch.randn(1, output_length, 2, device=device)  # Initial target for autoregressive
        outputs = model(inputs, tgt)

    return torch.cumsum(outputs[:, :output_length], dim=1)


def measure_standard_inference(model, inputs, tgt_dict, output_length, device):
    """Measure standard (non-autoregressive) inference time."""
    with torch.no_grad():
        fused_feature, pre_output = model.pre_forward(inputs)

        input_length = inputs.shape[1]
        remaining_length = max(0, output_length - input_length)

        if remaining_length > 0:
            # tgt = torch.randn(1, remaining_length, 2, device=device)
            tgt = tgt_dict[remaining_length]
            post_output = model(tgt, fused_feature, use_autoregressive=False)
            outputs = torch.cat([pre_output, post_output], dim=1)
        else:
            outputs = pre_output

    return torch.cumsum(outputs[:, :output_length], dim=1)

def measure_latency(model, model_type, inputs, output_length, device, num_runs=100):
    """Measure inference latency."""
    # Warmup
    tgt_dict = generate_tgt_dict(args, device)
    for _ in range(10):
        if model_type == "mlp":
            measure_mlp_inference(model, inputs, output_length, device)
        elif model_type == "autoregressive":
            measure_autoregressive_inference(model, inputs, output_length, device)
        elif model_type == "TDec":
            measure_pure_decoder_inference(model, inputs, output_length, device)
        else:  # standard
            measure_standard_inference(model, inputs, tgt_dict, output_length, device)

    # Actual timing

    torch.cuda.synchronize() if device.type == 'cuda' else None
    start = time.time()

    for _ in range(num_runs):
        if model_type == "mlp":
            measure_mlp_inference(model, inputs, output_length, device)
        elif model_type == "autoregressive":
            measure_autoregressive_inference(model, inputs, output_length, device)
        elif model_type == "TDec":
            measure_pure_decoder_inference(model, inputs, output_length, device)
        else:  # standard
            measure_standard_inference(model, inputs, tgt_dict, output_length, device)
        torch.cuda.synchronize() if device.type == 'cuda' else None

    end = time.time()
    return (end - start) / num_runs  # ms per run

def run_latency_tests(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    print(f"Using device: {device}")

    # Create sample input
    input_length = args.window_size
    sample_input = torch.randn(1, input_length, 3, device=device)  # batch_size=1, input_dim=3

    # Load models
    model_types = ["TDec", "mlp", "autoregressive", "standard"]
    models = {model_type: load_model(model_type, device) for model_type in model_types}

    # Test different output lengths
    output_lengths = range(args.min_output_length, args.max_output_length + 1, args.step_size)
    results = {model_type: [] for model_type in model_types}

    for model_type, model in models.items():
        for output_length in tqdm(output_lengths, desc="Testing output lengths"):
            latency = measure_latency(model, model_type, sample_input, output_length,
                                     device, args.num_runs)
            results[model_type].append(latency)
            print(f"{model_type} @ length {output_length}: {latency:.3f} ms")

    # Plot results
    plt.figure(figsize=(10, 6))
    for model_type, latencies in results.items():
        plt.plot(list(output_lengths), latencies, marker='o', label=model_type)

    plt.xlabel('Output Sequence Length')
    plt.ylabel('Inference Time (ms)')
    plt.title('Model Latency Comparison')
    plt.grid(True)
    plt.legend()
    plt.savefig(args.save_path)
    print(f"Plot saved to {args.save_path}")

    # Save numerical results to CSV
    df = pd.DataFrame({
        'output_length': list(output_lengths),
        **results
    })
    csv_path = args.save_path.replace('.png', '.csv')
    df.to_csv(csv_path, index=False)
    print(f"Data saved to {csv_path}")


def plot_latency(model_list=None):
    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    csv_path = "/home/user/wangtao/CLAM/Finger/latency_comparison_rtx4090.csv"
    lazy_map = {"mlp": "FTP (MLP)", "autoregressive": "FTP (AR)", "standard": "SAFTP",
                "TDec": "FTP (TDec)"}
    maker_map = {"mlp": "o", "autoregressive": "^", "standard": "*", "TDec": "s"}
    df = pd.read_csv(csv_path)

    nb_reduction = 1
    if model_list is None:
        model_list = ["mlp", "autoregressive", "standard"]

    output_lengths = df['output_length'][::nb_reduction]
    for i, model_name in enumerate(model_list):
        model_latencies = df[model_name][::nb_reduction]
        ax.plot(output_lengths, model_latencies, label=lazy_map[model_name], marker=maker_map[model_name], color=colors[i])

    ax.set_xlabel('Output Length')
    ax.set_ylabel('Latency (ms) with i5-13600KF')
    ax.set_yscale("log")
    ax.tick_params(axis='both', direction="in", which='major', labelsize=15, length=2, width=1)
    ax.tick_params(axis='both', direction="in", which='minor', length=2, width=1)  # Optional:  If you want minor ticks
    ax.set_xlim([0, 70])
    # ax.set_ylim([-0.003, 0.07])
    ax.legend(frameon=False,loc="upper left")
    plt.tight_layout()
    # plt.savefig('latency_i513600KF.png', dpi=300)
    plt.show()

def plot_latency_comparison():
    fig, ax = plt.subplots(figsize=(6.8, 4.8))

    # Load original smoothed data
    csv_path1 = "/home/user/wangtao/CLAM/Finger/latency_comparison_rtx4090.csv"
    df1 = pd.read_csv(csv_path1)
    nb_reduction = 2
    output_lengths1 = df1['output_length'][::nb_reduction]
    mlp_latencies1 = df1['mlp'][::nb_reduction]
    autoregressive_latencies1 = df1['autoregressive'][::nb_reduction]
    standard_latencies1 = df1['standard'][::nb_reduction]

    # Load i5 data
    csv_path2 = "/home/user/wangtao/CLAM/Finger/latency_comparison_i5.csv"
    df2 = pd.read_csv(csv_path2)
    output_lengths2 = df2['output_length'][::nb_reduction]
    mlp_latencies2 = df2['mlp'][::nb_reduction]
    autoregressive_latencies2 = df2['autoregressive'][::nb_reduction]
    standard_latencies2 = df2['standard'][::nb_reduction]

    # Plot original data with solid lines
    ax.plot(output_lengths1, mlp_latencies1, label='FTP (MLP) - GPU', marker='o', color=colors[0], linestyle='-')
    ax.plot(output_lengths1, autoregressive_latencies1, label='FTP (AR) - GPU', marker='^', color=colors[1], linestyle='-')
    ax.plot(output_lengths1, standard_latencies1, label='SAFTP - GPU', marker='*', color=colors[2], linestyle='-')

    # Plot i5 data with dashed lines
    ax.plot(output_lengths2, mlp_latencies2, label='FTP (MLP) - CPU', marker='o', color=colors[0], linestyle='--')
    ax.plot(output_lengths2, autoregressive_latencies2, label='FTP (AR) - CPU', marker='^', color=colors[1], linestyle='--')
    ax.plot(output_lengths2, standard_latencies2, label='SAFTP - CPU', marker='*', color=colors[2], linestyle='--')

    ax.set_xlabel('Output Length')
    ax.set_ylabel('Latency (ms)')
    ax.set_yscale("log")
    ax.tick_params(axis='both', direction="in", which='major', labelsize=15, length=2, width=1)
    ax.tick_params(axis='both', direction="in", which='minor', length=2, width=1)
    ax.set_xlim([0, 500])
    ax.set_ylim([0, 10])
    ax.set_yscale("log")
    ax.legend(frameon=False, ncol=2, loc="best")
    plt.tight_layout()
    plt.savefig('latency_comparison_combined_may5.png', dpi=500)
    plt.show()

def round_dict(dic):
    for key, value in dic.items():
        dic[key] = round(value, 2)
    return dic


def plot_stratify_by_length(saftp, ae, mlp, Tdec, threshold_k: int = 10, model_name_list=None):
    # copy from console output and it's can be easily reproduced

    # Default to all models if no specific list is provided
    if model_name_list is None:
        model_name_list = ['ae', 'mlp', 'tdec', 'saftp']

    # Filter models based on the provided list
    models = {
        'saftp': saftp,
        'ae': ae,
        'mlp': mlp,
        'tdec': Tdec
    }
    models = {name: models[name] for name in model_name_list if name in models}

    # Calculate average RMSE for each length group
    round_num = 1
    rmse_values = []
    for name, data in models.items():
        short = np.mean([v for k, v in data.items() if k < threshold_k]).round(round_num)
        long = np.mean([v for k, v in data.items() if k >= threshold_k]).round(round_num)
        avg = np.mean(list(data.values())).round(round_num)
        rmse_values.append([short, long, avg])

    # Transpose the data structure
    lazy_map = {"ae": "FTP (AR)", "saftp": "SAFTP", "mlp": "FTP (MLP)", "tdec": "FTP (TDec)"}
    labels_group = ['[3, {})'.format(threshold_k), r'[{}, 40]'.format(threshold_k), 'Average']
    labels_model = [lazy_map[name] for name in model_name_list]

    x = np.arange(len(labels_group))  # x-axis positions for sequence length groups
    width = 0.18  # the width of the bars

    fig, ax = plt.subplots(figsize=(6.8, 4.8))

    # Updated offsets for bars
    x_offsets = np.linspace(-width * (len(rmse_values) - 1) / 2, width * (len(rmse_values) - 1) / 2, len(rmse_values))

    for i, model_rmse in enumerate(rmse_values):
        positions = x + x_offsets[i]
        color_idx = i % len(colors)
        rects = ax.bar(positions, model_rmse, width, label=labels_model[i], color=colors[color_idx])
        autolabel(rects, ax)

    ax.set_xlabel("Input Sequence Length Groups")
    ax.set_ylabel('Average RMSE')
    ax.set_xticks(x)
    ax.set_xticklabels(labels_group)
    ax.legend(loc="upper center", ncol=4, frameon=False)
    ax.set_ylim(0, max([max(model) for model in rmse_values]) * 1.5)
    ax.tick_params(axis='both', direction="in", which='major', labelsize=15, length=2, width=1)
    fig.tight_layout()
    plt.savefig('comparison_by_model_may8.png', dpi=500)

    plt.show()

def autolabel(rects, ax): # Pass ax to autolabel function
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(round(height, 2)),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontname='Arial', fontsize=13)



if __name__ == "__main__":
    args = parse_args()
    seed_everything(42)
    model_list = ["mlp", "autoregressive", "TDec", "standard"]
    # note that the saftp_dict, ... can be generated by running the train_exp_dynamic.py or load from pickle
    # run_latency_tests(args)
    plot_latency(model_list=model_list)
    # plot_latency_comparison()
    # plot_stratify_by_length(saftp=saftp_dict, ae=ae_dict, mlp=mlp_dict, Tdec=Tdec_dict,
    #                         threshold_k=10, model_name_list=['ae', 'mlp', 'saftp'])  # Example call with threshold_k=10
