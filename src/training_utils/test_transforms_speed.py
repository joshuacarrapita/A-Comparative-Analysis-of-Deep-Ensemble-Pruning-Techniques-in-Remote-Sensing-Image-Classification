# system imports
import sys
import argparse
import time
import random
import inspect

# 3rd party imports
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# local imports
from configilm.extra.DataSets import BENv2_DataSet
from configilm.extra.BENv2_utils import band_combi_to_mean_std
from configilm.extra.CustomTorchClasses import MyGaussianNoise
from configilm.extra.CustomTorchClasses import MyRotateTransform
from configilm.extra._defaults import default_train_transform
from configilm.extra._defaults import default_train_transform_with_noise

# helper functions
def define_params(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_lmdb', type=str, required=True, help='Input path for lmdb file.')
    parser.add_argument('--metadata', type=str, required=True, help='Input path for metadata parquet.')
    parser.add_argument('--max_len', type=int, default='-1', help='Max number of images to process.')
    parser.add_argument('--seed', type=int, default='666', help='Random seed for RNG.')
    parser.add_argument('--channels', type=int, default='10', help='Number of channels the images have.')
    parser.add_argument('--outpath', type=str, required=True, help='Export path for results.')
    parser.add_argument('--v2', action='store_true', help='Use transforms v2.')
    parser.add_argument('--plot_frequency', type=int, default='0', help='Frequency with which to plot images. 0 = no plot, 1 = every image, 2 = every 2nd image etc.')
    parser.add_argument('--max_plot_per_transform', type=int, default='-1', help='Maximum amount of images to plot per transform. -1 = inf, 0 = no plot, 1 = 1 image, etc.')

    return parser.parse_args(args)

def set_seed(seed: int):
    random.seed(seed)

def transform_and_plot_images(dataset: BENv2_DataSet.BENv2DataSet = None, max_plot_per_transform: int = -1, plot_frequency: int = 1) -> float:
    # setup plot counter and queue
    plot_queue = []
    curr_plot_nu = 0
    if max_plot_per_transform < 0:
        max_plot_nu = len(dataset.patches)
    else:
        max_plot_nu = max_plot_per_transform

    start = time.time()

    # do transformations and save transformed images for plotting
    for idx in range(len(dataset.patches)):
        img, lbl, key = dataset.__getitem__(idx)

        if plot_frequency > 0 and (idx % plot_frequency) == 0 and curr_plot_nu < max_plot_nu:
            plot_queue.append((img, lbl, key))
            curr_plot_nu += 1
    
    end = time.time()
    runtime = end - start

    # plot images
    for idx, item in enumerate(plot_queue):
        # convert image to rgb and undo normalization
        img_rgb = item[0][:3, :, :]
        lbl = item[1]
        key = item[2] 

        img_plot = img_rgb.permute(1,2,0)
        img_plot = (img_plot - img_plot.min()) / (img_plot.max() - img_plot.min())
        
        # plot image
        plt.imshow(img_plot)
        plt.gcf().canvas.manager.set_window_title(f'{key}')
        plt.title(f'{lbl}')
        plt.show()

    return runtime

# main execution loop
def main(args):
    # setup environment
    assert Path(args.images_lmdb).resolve().is_dir(), f"Input directory does not exist or is not a directory: {args.images_lmdb}"
    assert Path(args.metadata).resolve().is_file(), f"Input file does not exist or is not a file: {args.metadata}"
    assert Path(args.outpath).resolve().is_dir(), f"Output directory does not exist or is not a directory: {args.outpath}"

    set_seed(args.seed)

    if args.v2:
        from torchvision.transforms import v2
        T = v2
        print(f'Using torchvision.transforms (v2): {T.__name__}')
    else:
        import torchvision.transforms as T
        print(f'Using torchvision.transforms (v1): {T.__name__}')

    data_dirs = {
            "images_lmdb": Path(args.images_lmdb).resolve(),
            "metadata_parquet": Path(args.metadata).resolve(),
            "metadata_snow_cloud_parquet": None,
            }
        
    ben_mean, ben_std = band_combi_to_mean_std(args.channels)
    ben_mean_list = ben_mean.tolist()
    img_size = 120
    times = []
    transforms = []


    # setup transforms 
    potential_transforms = [
        MyGaussianNoise(sigma=20.0),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        MyRotateTransform([0, 90, 180, 270]),
        T.RandomRotation(degrees=180, fill=ben_mean_list),
        T.RandomResizedCrop(size=(img_size,img_size),scale=(0.5,1.0),antialias=True),
        T.RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=ben_mean_list),
        T.ElasticTransform(alpha=50.0, sigma=5.0, fill=ben_mean_list),
        T.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9,1.1), fill=ben_mean_list),
        T.RandomPerspective(distortion_scale=0.25,p=0.5,fill=ben_mean_list),
        T.RandomCrop((img_size,img_size),fill=ben_mean_list),
        T.GaussianBlur(kernel_size=(3,3),sigma=(0.1,1.5)), 
    ]

    if args.v2:
        v2_only_transforms = [
            v2.RandomChannelPermutation(),
        ]

        potential_transforms.extend(v2_only_transforms)

    default_transforms = [default_train_transform(img_size, ben_mean, ben_std), default_train_transform_with_noise(img_size, ben_mean, ben_std, 20.0)]

    # execute transform loop
    # first for default_transforms
    for idx, default_transform in enumerate(default_transforms):

        if idx == 0: transforms.append('default_train_transform')
        if idx == 1: transforms.append('default_train_transform_with_noise')

        benchmark_ds = BENv2_DataSet.BENv2DataSet(
            data_dirs = data_dirs,
            split = 'train',
            bagging = False,
            seed = args.seed,
            transform = default_transform,
            max_len = args.max_len,
            img_size = (args.channels, img_size, img_size),
            return_extras = True,
        )

        print(f'Performing {default_transform} ...')

        runtime = transform_and_plot_images(dataset=benchmark_ds, max_plot_per_transform=args.max_plot_per_transform, plot_frequency=args.plot_frequency)
        
        times.append(runtime)

    # then for each potential_transform
    for transform in potential_transforms:
        # Basic transforms that are almost always needed
        transforms.append(str(transform))

        final_transforms = [
            T.Resize((img_size, img_size), antialias=True),
            transform,
            T.Normalize(mean=ben_mean, std=ben_std),
        ]

        final_transforms = T.Compose(final_transforms)
 
        benchmark_ds = BENv2_DataSet.BENv2DataSet(
            data_dirs = data_dirs,
            split = 'train',
            bagging = False,
            seed = args.seed,
            transform = final_transforms,
            max_len = args.max_len,
            img_size = (args.channels, img_size, img_size),
            return_extras = True,
        )

        print(f'Performing {transform} ...')

        runtime = transform_and_plot_images(dataset=benchmark_ds, max_plot_per_transform=args.max_plot_per_transform, plot_frequency=args.plot_frequency)
        
        times.append(runtime)


    df = pd.DataFrame({'Transform' : transforms, 'Abs. Time': times})
    df['Rel. Time (%)'] = df['Abs. Time'].transform(lambda x: x / x.sum()) * 100
    df['Pred. Hrs. without this'] = 12.223 * (1 - df['Rel. Time (%)'] / 100)
    df['Pred. saved Hrs. without this'] = 12.223 * (df['Rel. Time (%)'] / 100)
    df['Ratio against all others (%)'] = (df['Abs. Time'] / (df['Abs. Time'].sum() - df['Abs. Time'])) * 100
    df['Ratio against Default (%)'] = (df['Abs. Time'] / df['Abs. Time'][0]) * 100
    df['Ratio against Default w/ Noise (%)'] = (df['Abs. Time'] / df['Abs. Time'][1]) * 100
    df['Slower than Default (%)'] = ((df['Abs. Time'] - df['Abs. Time'][0]) / df['Abs. Time'][0]) * 100
    df['Slower than Default w/ Noise (%)'] = ((df['Abs. Time'] - df['Abs. Time'][1]) / df['Abs. Time'][1]) * 100
    df['x-fold Speedup against Default'] = (df['Abs. Time'][0] / df['Abs. Time'])
    df['x-fold Speedup against Default w/ Noise'] = (df['Abs. Time'][1] / df['Abs. Time'])

    df = df.sort_values(by=['Abs. Time'], ascending=False)
    df = df.round(3)
    #outpath = Path(args.outpath).resolve() / f'out_{time.strftime("%d%m%y%H%M%S",time.localtime())}.html'
    outpath = Path(args.outpath).resolve() / f'out.html'
    df.to_html(outpath)

    print(f'Successfully saved transform test results to: {str(outpath)}')

if __name__ == "__main__":
    args = define_params(sys.argv[1:])
    main(args)
