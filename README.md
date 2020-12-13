# Spine Segmentation using U-Net

My individual project for CMPE 257 focuses on semantic segmentation, specifically segmenting the spine from posteroanterior X-rays. This repo contains the code I use to perform the research. It is tuned for execution on the SJSU HPC.

## Results

## Commentary

I was unable to get TensorFlow 2 to recognize the GPU available to it on the SJSU HPC, so I dropped down to TensorFlow 1.15.0.

I knew it was all working when I did:
module load python3/3.6.6
module load cuda/10.0

```shell
srun -p gpu --gres=gpu -n 1 -N 1 -c 2 --pty /bin/bash
python
>>> import tensorflow as tf
>>> tf.test.is_gpu_available()
...
True
```

## Run the App

After gaining access to the SJSU HPC and cloning this repo there, you can follow the steps below to reproduce the results.

First, load the required modules for running the code on a GPU node:

```shell
module load python3/3.6.6
module load cuda/10.0
source venv/bin/activate
```

Next, submit a batch job to train the model to perform spine segmentation and write results out to the data directory. Clear the predictions before each run to ensure you don't mix up results across multiple runs.

```shell
rm -rf ../data/training/masks/predictions
rm -rf ../data/test/masks/predictions
sbatch app.sh | awk 'NF>1{print $NF}' | xargs -i sh -c 'sleep 5; tail -f slurm-{}.out'
```

## Retrieve the Results

Once the batch job has completed, you can retrieve the results by running the following commands on your machine (take care to alter the `SLURM_JOB_NUMBER` to the one associated with your run):

```shell
SLURM_JOB_NUMBER=30550
mkdir -p ~/Downloads/spine-segmentation/results/$SLURM_JOB_NUMBER/training
mkdir -p ~/Downloads/spine-segmentation/results/$SLURM_JOB_NUMBER/test
scp $SJSU_ID@coe-hpc1.sjsu.edu:~/cmpe257/spine-segmentation/code/slurm-$SLURM_JOB_NUMBER.out ~/Downloads/spine-segmentation/results/$SLURM_JOB_NUMBER/slurm-$SLURM_JOB_NUMBER.txt
scp -r $SJSU_ID@coe-hpc1.sjsu.edu:~/cmpe257/spine-segmentation/data/training/masks/predictions/final ~/Downloads/spine-segmentation/results/$SLURM_JOB_NUMBER/training
scp -r $SJSU_ID@coe-hpc1.sjsu.edu:~/cmpe257/spine-segmentation/data/test/masks/predictions/final ~/Downloads/spine-segmentation/results/$SLURM_JOB_NUMBER/test
```