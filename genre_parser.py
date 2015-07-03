"""Tool to extract genre tracks from medleydb"""

import logging
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
import tempfile
import shutil

import medleydb as mdb


class InstrumentParser:
    def __enter__(self):
        self.tempdir = tempfile.mkdtemp()
        return self

    def __exit__(self, type, value, traceback):
        if self.tempdir:
            shutil.rmtree(self.tempdir)

    def get_tracks(self):
        multitrack_list = mdb.load_all_multitracks()

        for track in multitrack_list:
            if track.genre:
                track.file_path = track.mix_path
                track.label = track.genre
                yield track
