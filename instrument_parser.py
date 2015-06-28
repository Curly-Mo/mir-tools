"""Tool to extract instrument stems from medleydb"""

import logging
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
import os
from collections import defaultdict
import subprocess
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

    def get_valid_instruments(self, min_sources):
        """Get set of instruments with at least min_sources different sources"""
        logging.info('Determining valid instruments...\n')
        multitrack_list = mdb.load_all_multitracks()

        instrument_counts = defaultdict(lambda: 0)
        for track in multitrack_list:
            if not track.has_bleed:
                instruments = set()
                for stem in track.stems:
                    instruments.add(stem.instrument)
                for instrument in instruments:
                    instrument_counts[instrument] += 1
        logging.debug(str(instrument_counts))
        return {i for i in instrument_counts if instrument_counts[i] >= min_sources}

    def get_stems(self, min_sources=10, instruments=None, rm_silence=False, trim=None):
        if instruments:
            valid_instruments = instruments
        else:
            valid_instruments = self.get_valid_instruments(min_sources)
        logging.info('Valid instruments: ' + str(valid_instruments) + '\n')

        multitrack_list = mdb.load_all_multitracks()

        for track in multitrack_list:
            if not track.has_bleed:
                for stem in track.stems:
                    if stem.instrument in valid_instruments:
                        if rm_silence or trim:
                            ext = os.path.splitext(stem.file_path)[1]
                            dest = tempfile.mktemp(suffix=ext, dir=self.tempdir)
                            logging.info('Processing (sox): ' + str(stem.file_path))
                            sox_args = ['sox', stem.file_path, dest]
                            if rm_silence:
                                sox_args.extend(['silence', '1', '0.1', '0.1%', '-1', '0.1', '0.1%'])
                            if trim:
                                sox_args.extend(['trim', '0', str(trim)])
                            subprocess.Popen(sox_args, stderr=subprocess.PIPE)
                            stem.file_path = dest
                        stem.label = stem.instrument
                        yield stem
