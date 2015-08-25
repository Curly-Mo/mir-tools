#!/usr/bin/env python
"""Tool to extract instrument stems from medleydb"""

import logging
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
import os
from collections import defaultdict
import subprocess
import tempfile
import shutil
import argparse

import medleydb as mdb


def get_instance(name):
    for cls in TrackParser.__subclasses__():
        if cls.__name__.lower().startswith(name.lower()):
            return cls()
    return TrackParser()


class TrackParser(object):
    track_attributes = [
        'file_path',
        'label',
    ]

    def __enter__(self):
        self.tempdir = tempfile.mkdtemp()
        return self

    def __exit__(self, type, value, traceback):
        if self.tempdir:
            shutil.rmtree(self.tempdir)

    def track_to_dict(self, track):
        d = {}
        for attribute in self.track_attributes:
            d[attribute] = getattr(track, attribute)
        return d


class InstrumentParser(TrackParser):
    def __init__(self):
        super(InstrumentParser, self).__init__()

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
        logging.info(instrument_counts)
        return {i for i in instrument_counts if instrument_counts[i] >= min_sources}

    def get_tracks(self, min_sources=10, instruments=None, rm_silence=False, trim=None, count=None, **kwargs):
        if instruments:
            valid_instruments = instruments
        else:
            valid_instruments = self.get_valid_instruments(min_sources)
        logging.info('Valid instruments: ' + str(valid_instruments) + '\n')

        counts = defaultdict(lambda: 0)

        multitrack_list = mdb.load_all_multitracks()

        for track in multitrack_list:
            if not track.has_bleed:
                for stem in track.stems:
                    if stem.instrument in valid_instruments:
                        if count and counts[stem.instrument] >= count:
                            continue
                        counts[stem.instrument] += 1
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
                        yield self.track_to_dict(stem)

        logging.info(counts)


class GenreParser(TrackParser):
    def __init__(self):
        super(GenreParser, self).__init__()

    def get_valid_genres(self, min_sources):
        """Get set of genres with at least min_sources different sources"""
        logging.info('Determining valid genres...\n')
        multitrack_list = mdb.load_all_multitracks()

        genre_counts = defaultdict(lambda: 0)
        for track in multitrack_list:
            if track.genre:
                genre_counts[track.genre] += 1
        logging.info(genre_counts)
        return {g for g in genre_counts if genre_counts[g] >= min_sources}

    def get_tracks(self, min_sources=10, genres=None, count=None, **kwargs):
        if genres:
            valid_genres = genres
        else:
            valid_genres = self.get_valid_genres(min_sources)
        logging.info('Valid genres: ' + str(valid_genres) + '\n')

        counts = defaultdict(lambda: 0)

        multitrack_list = mdb.load_all_multitracks()

        for track in multitrack_list:
            if track.genre and track.genre in valid_genres:
                if count and counts[track.genre] >= count:
                    continue
                counts[track.genre] += 1

                track.file_path = track.mix_path
                track.label = track.genre
                yield self.track_to_dict(track)
        logging.info(counts)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Extract instrument information from medleydb")
    parser.add_argument('-m', '--min_sources', nargs='*', default=10,
                        help='Min sources required for instrument selection')
    parser.add_argument('-i', '--instruments', nargs='*', default=None,
                        help='List of instruments to extract')
    parser.add_argument('-r', '--rm_silence', action='store_true',
                        help='Remove silence from audio files')
    parser.add_argument('-t', '--trim', type=int, default=None,
                        help='Trim all audio files down to this length (in seconds)')
    args = parser.parse_args()

    with InstrumentParser() as instruments:
        valid_instruments = instruments.get_valid_instruments(args.min_sources)
        logging.info('Valid instruments: ' + str(valid_instruments) + '\n')
