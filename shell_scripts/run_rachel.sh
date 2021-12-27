GPU=0

CUDA_VISIBLE_DEVICES=$GPU /media/data/conda/dlinsley/envs/powerAIlab/bin/python src/process/rachel_complete_segment.py
# CUDA_VISIBLE_DEVICES=$GPU /media/data/conda/dlinsley/envs/powerAIlab/bin/python src/process/rachel_membranes.py
# CUDA_VISIBLE_DEVICES=$GPU /media/data/conda/dlinsley/envs/powerAIlab/bin/python src/process/rachel_inference.py
python src/postprocess/plot_rachels.py

