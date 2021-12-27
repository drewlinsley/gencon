# Install postgres for managing volume segmentations/curations
1. sudo apt update
2. sudo apt install postgresql postgresql-contrib
3. `cp db/credentials.py.template db/credentials.py` (then edit the new file).
4. `cp config.py.template config.py` (then edit the new file)

# Install python packages for segmenting and curating volumes
1. `sudo apt-get install libpq-dev python-dev`
2. `python setup.py install`
3. `pip install -r requirements.txt`

# Segment a volume
As a test case, we focus on a high-resolution retinal volume from Rachel Wong's lab. We want to use our models, which were trained on k0725 (Ding et al., 2016), a low-resolution volume of mouse retina.

We will segment this volume by doing the following.

1. Prepare the data in webknossos format with `python src/preprocess/prepare_rachel_data.py`. This will convert data into a common and easily readable format (wkw), and add the coordinates of volumes that need to be processed into the database. There is a stride on these coordinates so that there will be redundancy/overlap between segmented volumes.
2. Segment the membranes and individuate cells with `python src/process/inference_rachel.py`
- To make this work, we have to apply a simple correction to the images. Our volume has voxel sizes of 13,13,26. Their volume has voxel sizes of 5,5,40. We find that our model generalizes well -- despite the two volumes coming from different animals and microscopes -- when we simply equalize the voxel size. This is done on-line in the inference script, and segmentations are resized into the native format before they are saved.
3. Postprocess to merge segments across the x/y/z axes. We use the overlap between segmented volumes to propogate labels along these axes and produce long neurite segments. Run this step with `python `.
4. Upload data to webknossos with `python `.

These steps can be automated by running `bash process_rachel_data.sh`


# Access the DB
- psql <db_user> -h 127.0.0.1 -d <db_name>
- psql wong -h 127.0.0.1 -d wong

