while read p; do
  sbatch --job-name "$p" eval.slurm
done <names.txt
