#!/usr/bin/env python
import os
import sshtunnel
import argparse
import psycopg2
import psycopg2.extras
import psycopg2.extensions
from . import credentials
import numpy as np
from tqdm import tqdm
from config import Config
sshtunnel.DAEMON = True  # Prevent hanging process due to forward thread
main_config = Config()



class db(object):
    def __init__(self, config):
        """Init global variables."""
        self.status_message = False
        self.db_schema_file = os.path.join('db', 'db_schema.txt')
        # Pass config -> this class
        for k, v in list(config.items()):
            setattr(self, k, v)

    def __enter__(self):
        """Enter method."""
        try:
            if main_config.db_ssh_forward:
                forward = sshtunnel.SSHTunnelForwarder(
                    credentials.machine_credentials()['ssh_address'],
                    ssh_username=credentials.machine_credentials()['username'],
                    ssh_password=credentials.machine_credentials()['password'],
                    remote_bind_address=('127.0.0.1', 5432))
                forward.start()
                self.forward = forward
                self.pgsql_port = forward.local_bind_port
            else:
                self.forward = None
                self.pgsql_port = ''
            pgsql_string = credentials.postgresql_connection(
                str(self.pgsql_port))
            self.pgsql_string = pgsql_string
            self.conn = psycopg2.connect(**pgsql_string)
            self.conn.set_isolation_level(
                psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
            self.cur = self.conn.cursor(
                cursor_factory=psycopg2.extras.RealDictCursor)
        except Exception as e:
            self.close_db()
            if main_config.db_ssh_forward:
                self.forward.close()
            print(e)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit method."""
        if exc_type is not None:
            print(exc_type, exc_value, traceback)
            self.close_db(commit=False)
        else:
            self.close_db()
        if main_config.db_ssh_forward:
            self.forward.close()
        return self

    def close_db(self, commit=True):
        """Commit changes and exit the DB."""
        try:
            self.conn.commit()
        except:
            print("Couldnt commit.")
        try:
            self.cur.close()
        except:
            print("Couldnt close cur.")
        try:
            self.conn.close()
        except:
            print("Couldnt close conn.")

    def recreate_db(self):
        """Initialize the DB from the schema file."""
        db_schema = open(self.db_schema_file).read().splitlines()
        for s in db_schema:
            t = s.strip()
            if len(t):
                self.cur.execute(t)

    def return_status(
            self,
            label,
            throw_error=False):
        """
        General error handling and status of operations.
        ::
        label: a string of the SQL operation (e.g. 'INSERT').
        throw_error: if you'd like to terminate execution if an error.
        """
        if label in self.cur.statusmessage:
            print('Successful %s.' % label)
        else:
            if throw_error:
                raise RuntimeError('%s' % self.cur.statusmessag)
            else:
                'Encountered error during %s: %s.' % (
                    label, self.cur.statusmessage
                )

    def reset(self):
        """Reset in coordinate info."""
        self.cur.execute(
            """
            UPDATE
            coordinates SET
            is_processing=False,
            processed=False,
            start_date=NULL,
            end_date=NULL
            """
        )
        if self.status_message:
            self.return_status('RESET')

    def populate_db_with_all_coords_fast(self, values):
        """
        Add a combination of parameter_dict to the db.
        ::
        """
        psycopg2.extras.execute_values(
            self.cur,
            """
            INSERT INTO coordinates
            (
                x,
                y,
                z,
                is_processing,
                processed,
            )
            VALUES %s
            """,
            values)
        if self.status_message:
            self.return_status('INSERT')

    def populate_db_with_all_coords(self, namedict, experiment_link=False):
        """
        Add a combination of parameter_dict to the db.
        ::
        experiment_name: name of experiment to add
        experiment_link: linking a child (e.g. clickme) -> parent (ILSVRC12)
        """
        self.cur.executemany(
            """
            INSERT INTO coordinates
            (
                x,
                y,
                z,
                is_processing,
                processed
            )
            VALUES
            (
                %(x)s,
                %(y)s,
                %(z)s,
                %(is_processing)s,
                %(processed)s
            )
            """,
            namedict)
        if self.status_message:
            self.return_status('INSERT')

    def reserve_coordinate(self, x, y, z):
        """Set is_processing=True."""
        self.cur.execute(
            """
            UPDATE coordinates
            SET is_processing=TRUE, start_date='now()'
            WHERE x=%s AND y=%s AND z=%s""" % (x, y, z))
        if self.status_message:
            self.return_status('UPDATE')

    def finish_coordinate(self, x, y, z):
        """Set membrane processed=True."""
        self.cur.execute(
            """
            UPDATE coordinates
            SET processed=TRUE and is_processing=True
            WHERE x=%s AND y=%s AND z=%s""" % (x, y, z))
        if self.status_message:
            self.return_status('UPDATE')

    def select_coordinate(self, namedict):
        """Select coordinates."""
        self.cur.executemany(
            """
            SELECT *
            FROM coordinates
            WHERE  x=%(x)s AND y=%(y)s AND z=%(z)s
            """,
            namedict)
        if self.status_message:
            self.return_status('SELECT')
        if self.cur.description is None:
            return None
        else:
            return self.cur.fetchall()

    def get_coordinate(self, experiment=None, random=False):
        """After returning coordinate, set processing=True."""
        self.cur.execute(
            """
            UPDATE coordinates
            SET is_processing=TRUE, start_date='now()'
            WHERE _id=(
                SELECT _id
                FROM coordinates
                WHERE (processed=FALSE AND is_processing=FALSE)
                OR (processed=FALSE AND DATE_PART('day', start_date - 'now()') > 0)
                LIMIT 1)
            RETURNING *
            """)
        if self.status_message:
            self.return_status('SELECT')
        return self.cur.fetchone()

    def get_total_coordinates(self):
        """Return the count of coordinates."""
        self.cur.execute(
            """
            SELECT count(*)
            FROM coordinates
            """)
        if self.status_message:
            self.return_status('SELECT')
        return self.cur.fetchone()

    def get_finished_coordinates(self):
        """Return the count of finished coordinates."""
        self.cur.execute(
            """
            SELECT count(*)
            FROM coordinates
            WHERE processed=True
            """)
        if self.status_message:
            self.return_status('SELECT')
        return self.cur.fetchone()

    def get_coordinate_info(self):
        """Return the count of finished coordinates."""
        self.cur.execute(
            """
            SELECT *
            FROM coordinates
            """)
        if self.status_message:
            self.return_status('SELECT')
        return self.cur.fetchall()

    def get_main_coordinate_info(self):
        """Return the count of finished coordinates."""
        self.cur.execute(
            """
            SELECT *
            FROM coordinates
            WHERE processed=True
            """)
        if self.status_message:
            self.return_status('SELECT')
        return self.cur.fetchall()

    def reset_rows(self, rows):
        """Set membrane processed=True."""
        self.cur.executemany(
            """
            UPDATE coordinates
            SET processed=False
            WHERE _id=%(_id)s""", rows)
        if self.status_message:
            self.return_status('UPDATE')


def process_rows(rows):
    """Set these rows to be processed."""
    config = credentials.postgresql_connection()
    with db(config) as db_conn:
        db_conn.reset_rows(rows)
        db_conn.return_status('RESET')


def initialize_database():
    """Initialize and recreate the database."""
    config = credentials.postgresql_connection()
    with db(config) as db_conn:
        db_conn.recreate_db()
        db_conn.return_status('CREATE')


def reset_database():
    """Reset coordinate progress."""
    config = credentials.postgresql_connection()
    with db(config) as db_conn:
        db_conn.reset()
        db_conn.return_status('RESET')


def populate_db(coords, slow=True):
    """Add coordinates to DB."""
    config = credentials.postgresql_connection()
    with db(config) as db_conn:
        coord_dict = []
        for coord in tqdm(
                coords,
                total=len(coords),
                desc='Processing coordinates'):
            x, y, z = coord
            if slow:
                coord_dict += [{
                    'x': int(x),
                    'y': int(y),
                    'z': int(z),
                    'is_processing': False,
                    'processed': False,
                    'run_number': None,
                    'chain_id': None}]
            else:
                coord_dict += [
                    int(x),
                    int(y),
                    int(z),
                    False,
                    False,
                    False,
                    False,
                    None,
                    None]
        print('Populating DB (this will take a while...)')
        if slow:
            db_conn.populate_db_with_all_coords(coord_dict)
        else:
            raise NotImplementedError('Not working for some reason...')
            db_conn.populate_db_with_all_coords_fast(coord_dict)
        db_conn.return_status('CREATE')


def get_coordinate():
    """Grab next row from coordinate table."""
    config = credentials.postgresql_connection()
    with db(config) as db_conn:
        coordinate = db_conn.get_coordinate()
        db_conn.return_status('SELECT')
    return coordinate


def finish_coordinate(x, y, z):
    """Finish off the segmentation coordinate from coordinate table."""
    config = credentials.postgresql_connection()
    with db(config) as db_conn:
        db_conn.finish_coordinate(x=x, y=y, z=z)
        db_conn.return_status('UPDATE')


def get_progress(extent=[5, 5, 5]):
    """Get percent finished of the whole connectomics volume."""
    config = credentials.postgresql_connection()
    with db(config) as db_conn:
        total_segments = db_conn.get_total_coordinates()['count']
        finished_segments = db_conn.get_finished_coordinates()['count']
        finished_segments *= np.prod(extent)
        prop_finished = float(finished_segments) / float(total_segments)
        print(('Segmentation is {}% complete.'.format(prop_finished * 100)))
    return prop_finished


def pull_main_seg_coors():
    """Return the list of membrane coordinates."""
    config = credentials.postgresql_connection()
    with db(config) as db_conn:
        finished_segments = db_conn.get_main_coordinate_info()
    return finished_segments


def main(
        initialize_db):
    """Test the DB."""
    if initialize_db:
        print('Initializing database.')
        initialize_database()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--initialize",
        dest="initialize_db",
        action='store_true',
        help='Recreate your database.')
    args = parser.parse_args()
    main(**vars(args))

