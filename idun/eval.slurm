#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=50G
#SBATCH --gres=gpu:A100m80:1
#SBATCH --constraint=A100
#SBATCH --job-name="eval-560m-wiki-wiki"
#SBATCH --output="./logs/res-%x.out"
#SBATCH --mail-type=end
#SBATCH --mail-user=andrebw@stud.ntnu.no

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
uname -a
echo "============================================================================="
echo "Working div: $SLURM_SUBMIT_DIR"
echo "Job name: $SLURM_JOB_NAME"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes used: $SLURM_JOB_NODELIST"
echo "Num nodes: $SLURM_JOB_NUM_NODES nodes"
echo "Num CPUs on node: $SLURM_CPUS_ON_NODE cores"
echo "Num GPUs on node: $SLURM_GPUS_ON_NODE"
echo "CUDA visible devices: $CUDA_VISIBLE_DEVICES"
echo "Total of $SLURM_NTASKS cores"
echo "Starting job..."
echo "------------------------------"
echo ""

module purge
module load fosscuda/2020b
module load Anaconda3/2020.07

python ../models/eval.py

echo ""
echo "------------------------------"
echo "Job $SLURM_JOB_ID finished!"
