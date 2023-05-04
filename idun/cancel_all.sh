# Cancels all slurm jobs of current user
# Argument 1 is the state of the jobs to cancel
# "R" means all running jobs, "PD" means all pending jobs
# Defaults to "PD"
squeue -u $USER --format="%.18i" -h --states="${1:-PD}" | while read -r line ; do
	scancel $line
	echo "job $line canceled"
done
