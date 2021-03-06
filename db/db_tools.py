'''DB tools'''
import os
import argparse
import numpy as np
import pandas as pd
# from db import db
import db
from glob2 import glob


def main(
        init_db=False,
        reset_coordinates=False,
        reset_priority=False,
        reset_config=False,
        reset_table=False,
        populate_db=None,
        get_progress=False,
        berson_correction=True,
        priority_list=None):
    """Routines for adjusting the DB."""
    if init_db:
        # Create the DB from a schema file
        db.initialize_database()

    if reset_coordinates:
        # Create the DB from a schema file
        db.reset_database()

    if populate_db:
        # Fill the DB with a coordinates + global config
        import pdb;pdb.set_trace()
        print("Need to add coords here")
        assert config.coord_path is not None
        if os.path.exists(config.coord_path):
            coords = np.load(config.coord_path)
        else:
            print(
                'Gathering coordinates from: %s '
                '(this may take a while)' % config.coord_path)
            coords = glob(GLOB_MATCH)
            coords = [os.path.sep.join(
                x.split(os.path.sep)[:-1]) for x in coords]
            np.save(config.coord_path, coords)
        db.populate_db(coords)

    if reset_priority:
        # Create the DB from a schema file
        db.reset_priority()

    if reset_table:
        db.reset_a_table()

    if reset_config:
        # Create the global config to starting values
        db.reset_config()

    if get_progress:
        # Return the status of segmentation
        db.get_progress()

    if priority_list is not None:
        # Add coordinates to the DB priority list
        assert '.csv' in priority_list, 'Priorities must be a csv.'
        priorities = pd.read_csv(priority_list)
        if berson_correction:
            max_chain = db.get_max_chain_id() + 1
            chains = range(max_chain, max_chain + len(priorities))
            priorities.x //= 128
            priorities.y //= 128
            priorities.z //= 128
            priorities['prev_chain_idx'] = 0
            priorities['chain_id'] = chains
            priorities['processed'] = False
            priorities['force'] = True
            db.update_max_chain_id(np.max(chains))
        db.add_priorities(priorities)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--init_db',
        dest='init_db',
        action='store_true',
        help='Initialize DB from schema (must be run at least once).')
    parser.add_argument(
        '--populate_db',
        dest='populate_db',
        action='store_true',
        help='Add all coordinates to the DB.')
    parser.add_argument(
        '--reset_coordinates',
        dest='reset_coordinates',
        action='store_true',
        help='Reset coordinate progress.')
    parser.add_argument(
        '--reset_table',
        dest='reset_table',
        action='store_true',
        help='Reset status for a table.')
    parser.add_argument(
        '--reset_priority',
        dest='reset_priority',
        action='store_true',
        help='Reset coordinate progress.')
    parser.add_argument(
        '--reset_config',
        dest='reset_config',
        action='store_true',
        help='Reset global config.')
    parser.add_argument(
        '--get_progress', 
        dest='get_progress',
        action='store_true',
        help='Return proportion-finished of total segmentation.')
    parser.add_argument(
        '--berson_correction',
        dest='berson_correction',
        action='store_false',
        help='No berson coordinate correction.')
    parser.add_argument(
        '--priority_list',
        dest='priority_list',
        type=str,
        default=None,  # 'db/priorities.csv'
        help='Path of CSV with priority coordinates '
        '(see db/priorities.csv for example).')
    args = parser.parse_args()
    main(**vars(args))

