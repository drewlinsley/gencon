# pkill -fe loky
# /media/data/conda/dlinsley/envs/powerAIlab/bin/python parallel_cube_merged_wkv.py

# Delete server data
# touch /cifs/data/tserre/CLPS_Serre_Lab/connectomics/cubed_mag1/delete_me

# Delete the lrs cubes
rm -rf /cifs/data/tserre/CLPS_Serre_Lab/connectomics/cubed_mag1/merge_data_wkw_class/*

# Delete current cubes
rm -rf /gpfs/data/tserre/data/wkcube/merge_data_wkw_class

# Delete current skeletonization
rm -rf /gpfs/data/tserre/data/wkcube/skeleton_merge_class

# Delete current compresson
rm -rf /gpfs/data/tserre/data/wkcube_compress_class

# Run the cubing
mkdir /gpfs/data/tserre/data/wkcube/merge_data_wkw_class
# /media/data/conda/dlinsley/envs/wkcuber/bin/python parallel_cube_merged_wkv.py  # Gathers data
# /media/data/anaconda3-ibm/bin/python parallel_cube_merged_wkv.py
/media/data/anaconda3-ibm/bin/python glob_parallel_cube_merged_wkv.py

# Skeletonize
export PYTHONPATH=$PYTHONPATH:/users/dlinsley/wkcuber/
rm -rf /gpfs/data/tserre/data/wkcube/skeleton_merge_class
cp /cifs/data/tserre/CLPS_Serre_Lab/connectomics/cubed_mag1/pbtest/ding_class/datasource-properties.json /gpfs/data/tserre/data/wkcube/

# # /media/data/anaconda3-ibm/bin/python merge_skeltons_in_wkw.py --input=/gpfs/data/tserre/data/wkcube --layer_name=merge_data_wkw_class --nml=/users/dlinsley/ffn_membrane/All_Skels_to_Stitch_Segs.nml  --output=/gpfs/data/tserre/data/wkcube/skeleton_merge_class
# /media/data/anaconda3-ibm/bin/python merge_skeltons_in_wkw.py --input=/gpfs/data/tserre/data/wkcube --layer_name=merge_data_wkw_class --nml=/cifs/data/tserre/CLPS_Serre_Lab/connectomics/from_berson/trees_for_merge_9-2021/@RGCs_unique.nml   --output=/gpfs/data/tserre/data/wkcube/skeleton_merge_class

# Compress wkws
mkdir /gpfs/data/tserre/data/wkcube_compress_class
python -m wkcuber.compress --layer merge_data_wkw_class /gpfs/data/tserre/data/wkcube/ /gpfs/data/tserre/data/wkcube_compress_class
# python -m wkcuber.compress --layer merge_data_wkw_class /gpfs/data/tserre/data/wkcube/skeleton_merge_class /gpfs/data/tserre/data/wkcube_compress_class

# Generate multiple scales
# cp /cifs/data/tserre/CLPS_Serre_Lab/connectomics/cubed_mag1/pbtest/ding_class/datasource-properties.json /gpfs/data/tserre/data/wkcube_compress_class/
# python -m wkcuber.downsampling --layer_name merge_data_wkw_class /gpfs/data/tserre/data/wkcube_compress_class --from_mag 1 --jobs 16 --max 16 python -m wkcuber.downsampling
# mv /gpfs/data/tserre/data/wkcube_compress_class/merge_data_wkw_class/2-2-1 /gpfs/data/tserre/data/wkcube_compress_class/merge_data_wkw_class/2
# mv /gpfs/data/tserre/data/wkcube_compress_class/merge_data_wkw_class/4-4-2 /gpfs/data/tserre/data/wkcube_compress_class/merge_data_wkw_class/4
# mv /gpfs/data/tserre/data/wkcube_compress_class/merge_data_wkw_class/8-8-4 /gpfs/data/tserre/data/wkcube_compress_class/merge_data_wkw_class/8
# mv /gpfs/data/tserre/data/wkcube_compress_class/merge_data_wkw_class/16-16-8 /gpfs/data/tserre/data/wkcube_compress_class/merge_data_wkw_class/16

# Mv the compressed cubes
# rsync -rzva -O --progress /gpfs/data/tserre/data/wkcube_compress/merge_data_wkw/* /cifs/data/tserre/CLPS_Serre_Lab/connectomics/cubed_mag1/merge_data_wkw/
msrsync -P -p 32 --stats --rsync "-rvz --no-perms" /gpfs/data/tserre/data/wkcube_compress_class/merge_data_wkw_class/* /cifs/data/tserre/CLPS_Serre_Lab/connectomics/cubed_mag1/merge_data_wkw_class/

# And sync
touch /cifs/data/tserre/CLPS_Serre_Lab/connectomics/cubed_mag1/sync_me

# msrsync -P -p 32 --stats --rsync "-rvz --no-perms" /localscratch/wong_1/ /cifs/data/tserre/CLPS_Serre_Lab/connectomics/from_berson/

