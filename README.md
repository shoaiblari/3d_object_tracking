# 3d_object_tracking
3D Object Tracking for Autonomous Driving

# Build

## Setup
If your base github directory is ~/w, then proceed as follows.

```
 cd ~/w
```

Clone the repos.
```
git clone https://github.com/shoaiblari/3d_object_tracking.git
```

Clone the nuscenes-devkit
```
git clone https://github.com/nutonomy/nuscenes-devkit.git
```

Install the nuscenes dataset in directory such as `/aa273/data/nuscenes`.

Make sure have anaconda3 installed.

Your `PATH` should have anaconda3/bin on it, and your PYTHONPATH should include
the nuscenes-devkit. In the example below, the following two lines were added to
the end of the .bashrc to make this happen.

```
export PATH=/home/ubuntu/anaconda3/bin:$PATH
export PYTHONPATH="/home/slari1/w/nuscenes-devkit/python-sdk:${PYTHONPATH}"
```
logout and log back in.

Run
```
conda init bash
```

Edit your .bashrc to add the following line at the end.
```
conda deactivate
```

Logout and log back in.

Create the `pt` Conda environment.

```
conda create --name pt python=3.6
conda activate pt
```

Install the nuscenes-devkit
```
cd ~/w/nuscenes-devkit
pip install -r setup/requirements.txt

pip install -U llvmlite==0.32.1
```


Install required packages for stereo-based-tracking.
```
cd ~/w/3d_object_tracking
pip install -r requirements.txt
```

## Test Run
Once the above installs succeed, change the directory and file locations in the following files to match your environment.

- Download nuscenes trainval meta data and the File blobs of 85 scenes, part 1, if you want to analyze the data.  
- In 'data_files.py' change all directory and file locations to match where the detection and data files are located.

After making these changes, run the main program by.

```
cd ~/w/3d_object_tracking
python main.py val 2 m 11 greedy true nuscenes results/000008;
```

Upon successful run, tracking results will be in the `~/w/3d_object_tracking/results/000008/val/results_val_probabilistic_tracking.json` file.

## To generate aggregate results
Run get_nuscenes_stats.py --output_file results/covariance/covariance_results

## To generate results on a per-frame basis
Run evaluate_nuscenes.py --output_dir results/results_000008 results_val_probabilistic_tracking.json

## To plot the bounding boxes per object
Run plotting3d.py

## To generate the covariance matrices
Run get_nuscenes_stats.py --output_file results/covariance/covariance_results
