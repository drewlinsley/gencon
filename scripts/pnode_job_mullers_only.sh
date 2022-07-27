
read -p "Enter the GPU to run on: " GPU

CONFIG="configs/W-Q.yml"
CONFIG="configs/GJD2.yml"
PROGRESS=1000
while [ $PROGRESS -ge 0 ];
do
    PROGRESS=$(python db/db.py --check_progress)
    echo "$PROGRESS jobs left"

    # Run job here
    echo $GPU
    CUDA_VISIBLE_DEVICES=$GPU python src/process/muller_segment_routine.py $CONFIG
done

