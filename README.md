# Optional: Install postgres for managing volume segmentations/curations
1. sudo apt update
2. sudo apt install postgresql postgresql-contrib
3. `cp db/credentials.py.template db/credentials.py` (then edit the new file).
4. `cp config.py.template config.py` (then edit the new file)

# Optional: Create database and add a couple of dev packages for python
1. `sudo apt-get install libpq-dev python-dev`
2. `python setup.py install`

# Install python packages for segmenting and curating volumes
1. `pip install -r requirements.txt`

# Segment a volume
As a test case, we focus on a high-resolution retinal volume from Rachel Wong's lab. We want to use our models, which were trained on k0725 (Ding et al., 2016), a low-resolution volume of mouse retina.

We will segment this volume by doing the following.

(Optional) Prepare the data in webknossos format with `python src/preprocess/prepare_rachel_data.py`. This will convert data into a common and easily readable format (wkw), and add the coordinates of volumes that need to be processed into the database. There is a stride on these coordinates so that there will be redundancy/overlap between segmented volumes.

# Run membrane and neurite segmentation on the data specified in your config file:

1. python src/process/segment_routine.py configs/<config_name.yml>
- To make this work, we have to apply a simple correction to the images. Our volume has voxel sizes of 13,13,26. Their volume has voxel sizes of 5,5,40. We find that our model generalizes well -- despite the two volumes coming from different animals and microscopes -- when we simply equalize the voxel size. This is done on-line in the inference script, and segmentations are resized into the native format before they are saved.

# Automate the segmentation process
1. bash pnode_job_full_segmentation.sh
2. bash pnode_job_waterz_postproc.sh

# (If you have multiple subvolumes that need to be merged, merge them into a big volume)
1. python src/postprocess/combine_merge_and_upload_full.py <config name>

# Upload to WK
1. python src/postprocess/upload_john.py <config name>

These steps can be automated in a bash script by adding them sequentially. See `bash process_rachel_data.sh` for an example.

# Uploads
If this fails due to an os_error, try `ulimit -n 2048`

# Access the DB
- psql <db_user> -h 127.0.0.1 -d <db_name>
- psql wong -h 127.0.0.1 -d wong

# TODO:
- Change config class to yml files for flexibility.
- Right now I'm symlinking the gcp models in with ln -s /media/data_cifs/projects/prj_connectomics/gcp_models/src/pl_modules/ src/gcp_models
  Turn these into pip install packages then import those models instead
