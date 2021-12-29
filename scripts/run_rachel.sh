GPU=0

export PYTHONPATH=$PYTHONPATH:$(pwd)
CUDA_VISIBLE_DEVICES=$GPU python src/process/segment_routine.py configs/W-Q.yml

