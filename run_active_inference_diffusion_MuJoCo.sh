#!/bin/bash
#SBATCH --job-name=ActInfDiff
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=80G  
#SBATCH --time=1-12:59:00
#SBATCH --account=def-irina
#SBATCH --output=/home/memole/projects/def-irina/memole/logs/run_active-inference-diffusion-cheetah-seed-1_%N-%j.out
#SBATCH --error=/home/memole/projects/def-irina/memole/logs/run_active-inference-diffusion-cheetah-seed-1_%N-%j.err
#SBATCH --mail-user=sheikhbahaee@gmail.com              # notification for job conditions
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

module load StdEnv/2023
module load gcc/12.3
module load cuda/12.6
module load python/3.11
module load scipy-stack/2024a
module load arrow/17.0.0
module load mujoco
module load openmpi
module load mpi4py/3.1.6
module load opencv/4.9.0
module load imkl/2023.2.0
module load rust/1.70.0
module load cmake
DIR=/home/memole/projects/def-irina/memole/active-inference-diffusion

unset PYOPENGL_PLATFORM
# Or explicitly set it to osmesa
export PYOPENGL_PLATFORM=egl
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=${CUDA_VISIBLE_DEVICES:-0}
#virtualenv --no-download --clear /home/memole/ActInfDiffEnv
source /home/memole/ActInfDiffEnv/bin/activate


CURRENT_PATH=`pwd`
echo "current path ---> $CURRENT_PATH"
pip install --upgrade pip setuptools wheel
#pip install --no-index --no-cache-dir numpy 
#pip install --no-index torch torchvision torchtext torchaudio
#pip install --no-index --no-cache-dir wandb
#pip install --no-cache-dir -r ~/projects/def-irina/memole/active-inference-diffusion/requirements.txt

wandb login a2a1bab96ebbc3869c65e3632485e02fcae9cc42

CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline WANDB_START_METHOD=thread python -m examples.train_mujoco --env HalfCheetah-v4 --pixels --seed 42 --num_parallel_envs 15 --timesteps 2000000
