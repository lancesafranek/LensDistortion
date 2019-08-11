import json
import pprint
import exifread
import argparse


__version__ = '0.0.2'
__author__ = 'Stefan Safranek'


def get_metadata(filename):
    ''' Reads EXIF tags from image file and returns metadata.
        @method: getMetadata
        @params: filename {str}
        @return: {dict}
    '''

    metadata = {}
    with open(filename, 'rb') as fh:
        tags = exifread.process_file(fh)
        for tag in tags.keys():
            if tag not in ['JPEGThumbnail', 'TIFFThumbnail', 'EXIF MakerNote']:
                parts = tag.split(' ')
                section = parts[0]
                prop = '_'.join(parts[1:])
                if section not in metadata:
                    metadata[section] = {}
                metadata[section][prop] = tags[tag].printable
    return metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image', help='image file to extract EXIF data from', required=True)
    args = parser.parse_args()
    metadata = getMetadata(args.image)
    print(json.dumps(metadata))
