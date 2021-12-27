import os
import webknossos as wk
from src.process import rachel_membranes
from src.process import rachel_inference



def main(
    ds_input="/localscratch/wong_1",
    ds_layer="color",
    image_shape=[280, 4740, 4740],
    resize_mod=[1, 4, 4]):

    # next_coordinate = db.get_next_segmentation_coordinate()
    next_coordinate = {"x": 0, "y": 0, "z": 0}
    if next_coordinate is None:
        # No need to process this point
        logging.exception('No more coordinates found!')
        return
    x, y, z = next_coordinate['x'], next_coordinate['y'], next_coordinate['z']  # noqa

    # Get images
    import pdb;pdb.set_trace()
    ds = wk.Dataset(os.path.join(ds_input, ds_layer, "1"))
    cube_in = ds.read((x, y, z), image_shape)[0]


    # Resize images

    # Segment membranes

    # Segment neurites

    print("Finished")


if __name__ == '__main__':
    main()
