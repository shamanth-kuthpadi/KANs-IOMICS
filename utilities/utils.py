import torch
import os
import copy
import numpy as np
import moviepy.video.io.ImageSequenceClip # moviepy == 1.0.3
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import glob
from sklearn.datasets import make_moons
import json


def create_dataset(device, train_num=500, test_num=500):

    def generate_contrastive(x):
        # positive samples
        batch = x.shape[0]
        x[:,2] = torch.exp(torch.sin(torch.pi*x[:,0])+x[:,1]**2)
        x[:,3] = x[:,4]**3

        # negative samples
        def corrupt(tensor):
            y = copy.deepcopy(tensor)
            for i in range(y.shape[1]):
                y[:,i] = y[:,i][torch.randperm(y.shape[0])]
            return y

        x_cor = corrupt(x)
        x = torch.cat([x, x_cor], dim=0)
        y = torch.cat([torch.ones(batch,), torch.zeros(batch,)], dim=0)[:,None]
        return x, y

    x = torch.rand(train_num, 6) * 2 - 1
    x_train, y_train = generate_contrastive(x)

    x = torch.rand(test_num, 6) * 2 - 1
    x_test, y_test = generate_contrastive(x)

    dataset = {}
    dataset['train_input'] = x_train.to(device)
    dataset['test_input'] = x_test.to(device)
    dataset['train_label'] = y_train.to(device)
    dataset['test_label'] = y_test.to(device)
    return dataset


def create_video(video_name, input_dir, output_dir, fps=5):
    
    files = os.listdir(input_dir)
    train_index = []
    for file in files:
        if file[0].isdigit() and file.endswith('.jpg'):
            train_index.append(int(file[:-4]))

    train_index = np.sort(train_index)

    image_files = []
    for i in train_index:
        image_files.append(input_dir+'/'+str(i)+'.jpg')
    
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)

    output_path = os.path.join(output_dir if output_dir else ".", video_name + ".mp4")
    clip.write_videofile(output_path)

    subprocess.call(('open', output_path))

def plot_results(file):
    df = pd.read_csv(file)

    df.set_index('step', inplace=True)

    metrics = df.columns.tolist()
    n = len(metrics)
    cols = 5
    rows = int(np.ceil(n/cols))

    fig, axes = plt.subplots(rows, cols, figsize=(15, rows*3), sharex=True)
    axes = axes.flatten()

    for ax, metric in zip(axes, metrics):
        ax.plot(df.index, df[metric], lw=1.5)
        ax.set_title(metric)
        ax.grid(True)

    for ax in axes[n:]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()

def plot_multiple_results(files, 
                          labels=None, 
                          cols=3, 
                          figsize_per_row=(15, 3),
                          colors=None,
                          save_dir=None,
                          save_name="plot.png"):

    if isinstance(files, str):
        files = glob.glob(files)
    n_runs = len(files)
    if labels is None:
        labels = [os.path.splitext(os.path.basename(f))[0] for f in files]
    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    dfs = [pd.read_csv(f).set_index('step') for f in files]

    metrics = dfs[0].columns.tolist()
    n_metrics = len(metrics)
    rows = int(np.ceil(n_metrics / cols))

    fig, axes = plt.subplots(rows, cols, 
                             figsize=(figsize_per_row[0], rows*figsize_per_row[1]), 
                             sharex=True)
    axes = axes.flatten()

    for ax, metric in zip(axes, metrics):
        for i, df in enumerate(dfs):
            ax.plot(df.index, df[metric], 
                    lw=1.5, 
                    label=labels[i], 
                    color=colors[i % len(colors)])
        ax.set_title(metric)
        ax.grid(True)
        ax.legend(fontsize='small', loc='best')

    for ax in axes[n_metrics:]:
        ax.axis('off')

    plt.tight_layout()

    # Save if a directory is provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, save_name)
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to: {save_path}")

    plt.show()


def create_dataset_super(device, seed=147):
    dataset = {}
    train_input, train_label = make_moons(n_samples=1000, shuffle=True, noise=0.1, random_state=seed)
    test_input, test_label = make_moons(n_samples=1000, shuffle=True, noise=0.1, random_state=seed)

    dtype = torch.get_default_dtype()
    dataset['train_input'] = torch.from_numpy(train_input).type(dtype).to(device)
    dataset['test_input'] = torch.from_numpy(test_input).type(dtype).to(device)
    dataset['train_label'] = torch.from_numpy(train_label[:,None]).type(dtype).to(device)
    dataset['test_label'] = torch.from_numpy(test_label[:,None]).type(dtype).to(device)

    X = dataset['train_input']
    y = dataset['train_label']

    return dataset

def create_dataset_clas(device, seed=147):
    dataset = {}
    train_input, train_label = make_moons(n_samples=1000, shuffle=True, noise=0.1, random_state=13)
    test_input, test_label = make_moons(n_samples=1000, shuffle=True, noise=0.1, random_state=15)

    dtype = torch.get_default_dtype()
    dataset['train_input'] = torch.from_numpy(train_input).type(dtype).to(device)
    dataset['test_input'] = torch.from_numpy(test_input).type(dtype).to(device)
    dataset['train_label'] = torch.from_numpy(train_label).type(torch.long).to(device)
    dataset['test_label'] = torch.from_numpy(test_label).type(torch.long).to(device)

    X = dataset['train_input']
    y = dataset['train_label']

    return dataset

def export_metrics_to_json(files, output_dir="json_output"):
    if isinstance(files, str):
        files = glob.glob(files)
    os.makedirs(output_dir, exist_ok=True)

    dfs = [pd.read_csv(f).set_index('step') for f in files]
    labels = [os.path.splitext(os.path.basename(f))[0] for f in files]
    metrics = dfs[0].columns.tolist()

    all_json = {}

    for metric in metrics:
        metric_data = []

        for label, df in zip(labels, dfs):
            data_points = [
                {"x": int(step), "y": float(df.loc[step, metric])}
                for step in df.index
            ]
            metric_data.append({
                "label": label,
                "data": data_points
            })

        json_object = {
            "title": metric,
            "data": metric_data
        }

        out_path = os.path.join(output_dir, f"{metric}.json")
        with open(out_path, "w") as f:
            json.dump(json_object, f, indent=2)
        all_json[metric] = json_object

    print(f"Exported {len(metrics)} metric JSON files to '{output_dir}/'")
    return all_json