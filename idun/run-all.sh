while read p; do
  sbatch --job-name "$p" 2job.slurm
done <names.txt
