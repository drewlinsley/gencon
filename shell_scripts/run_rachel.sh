GPU=0

CUDA_VISIBLE_DEVICES=$GPU /media/data/conda/dlinsley/envs/powerAIlab/bin/python get_rachel_membranes.py
CUDA_VISIBLE_DEVICES=$GPU /media/data/conda/dlinsley/envs/powerAIlab/bin/python rachel_inference.py
python plot_rachels.py

