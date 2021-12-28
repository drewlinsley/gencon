GPU=0

export PYTHONPATH=$PYTHONPATH:$(pwd)
CUDA_VISIBLE_DEVICES=$GPU python src/process/segment_routine.py
# CUDA_VISIBLE_DEVICES=$GPU /media/data/conda/dlinsley/envs/powerAIlab/bin/python src/process/rachel_membranes.py
# CUDA_VISIBLE_DEVICES=$GPU /media/data/conda/dlinsley/envs/powerAIlab/bin/python src/process/rachel_inference.py
# python src/postprocess/plot_rachels.py

