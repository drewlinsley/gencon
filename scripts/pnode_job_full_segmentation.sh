read -p "Enter the GPU to run on: " GPU

CONFIG="configs/john_ball_test.yml"
PROGRESS=1000
while [ $PROGRESS -ge 0 ];
do
    PROGRESS=$(python db/db.py --check_progress)
    echo "$PROGRESS jobs left"

    # Run job here
    echo $GPU
    CUDA_VISIBLE_DEVICES=$GPU python src/process/segment_routine.py $CONFIG
done

