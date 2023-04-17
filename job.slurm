#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --job-name="IDATT2900-072 MGT-Detection"
#SBATCH --output=idun-results.out
#SBATCH --mail-user=andrebw@stud.ntnu.no
#SBATCH --mail-type=ALL

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
uname -a
echo "============================================================================="
echo "Working div: $SLURM_SUBMIT_DIR"
echo "Job name: $SLURM_JOB_NAME"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes used: $SLURM_JOB_NODELIST"
echo "Num nodes: $SLURM_JOB_NUM_NODES nodes"
echo "Num CPU cores: $SLURM_CPUS_ON_NODE cores"
echo "Num GPUs: $SLURM_GPUS GPUs"
echo "Mem per CPU: $SLURM_MEM_PER_CPU"
echo "Mem per GPU: $SLURM_MEM_PER_GPU"
echo "CUDA visible devices: $CUDA_VISIBLE_DEVICES"
echo "Cores per node: $SLURM_CPUS_ON_NODE"
echo "Total of $SLURM_NTASKS cores"
echo "Starting job..."
echo "------------------------------"
echo ""

module purge
module load fosscuda/2020b
module load Anaconda3/2020.07

pip install -r requirements.txt

python main.py

echo ""
echo "------------------------------"
echo "Job finished!"
