import json
import pprint
import exifread
import argparse
# from PIL import Image
# from PIL import ExifTags
# from PIL import TiffTags


# def _sanitize(metadata):
#     ''' converts datatypes of image metadata to json serializable types.
#         @method: _sanitize
#         @params: metadata {dict}
#         @return: {dict}
#     '''
#     for k in metadata:
#         if tuple == type(metadata[k]):
#             metadata[k] = list(metadata[k])
#         elif bytes == type(metadata[k]):
#             metadata[k] = metadata[k].decode('utf-8')
#         elif dict == type(metadata[k]):
#             metadata[k] = _sanitize(metadata[k])
#         else:
#             print(type(metadata[k]))
#     return metadata


def getMetadata(filename):
    ''' Reads EXIF tags from image file and returns metadata.
        @method: getMetadata
        @params: filename {str}
        @return: {dict}
    '''

    metadata = {}
    with open(filename, 'rb') as fh:
        tags = exifread.process_file(fh)
        for tag in tags.keys():
            if tag not in ['JPEGThumbnail', 'TIFFThumbnail', 'EX?IF MakerNote']:
                parts = tag.split(' ')
                section = parts[0]
                prop = '_'.join(parts[1:])
                if section not in metadata:
                    metadata[section] = {}
                metadata[section][prop] = tags[tag].printable
                # metadata[tag] = tags[tag].printable
    return metadata


    # img = Image.open(filename)
    #
    # if 'PNG' == img.format:
    #     if 'exif' in img.info.keys():
    #         metadata = {
    #             ExifTags.TAGS[k]: v for k, v in img._getexif().items() if k in ExifTags.TAGS
    #         }
    #         if 'GPSInfo' in metadata:
    #             gpsinfo = {}
    #             for key in metadata['GPSInfo'].keys():
    #                 decode = ExifTags.GPSTAGS.get(key, key)
    #                 gpsinfo[decode] = metadata['GPSInfo'][key]
    #             metadata['GPSInfo'] = gpsinfo
    # elif 'JPEG' == img.format:
    #     metadata = {'TODO': True}
    # elif 'TIFF' == img.format:
    #     # metadata = {
    #     #     TiffTags.TAGS[k]: img.tag[k] for k in img.tag
    #     # }
    #     print('TODO')
    # else:
    #     print(img.format)
    #
    # return _sanitize(metadata)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image', help='image file to extract EXIF data from', required=True)
    args = parser.parse_args()
    metadata = getMetadata(args.image)
    print(json.dumps(metadata))
