while read p; do
  sbatch --job-name "$p" auto-job.slurm
done <names.txt
