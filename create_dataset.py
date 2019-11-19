"""Create TFRecord from geojson """

import os
import json
import argparse
import math
from collections import namedtuple

import numpy as np
import cv2
from tqdm import tqdm
from skimage.io import imread
from shapely.geometry import Polygon
import tensorflow as tf

# Object Class
Object = namedtuple('Object', 'coord cls_idx cls_text')

# Pacth Class
Patch = namedtuple('Patch', 'image_id image row col objects')


def get_patch_image(image, row, col, patch_size):
    patch_image_height = patch_size if image.shape[0] - row > patch_size else image.shape[0] - row
    patch_image_width = patch_size if image.shape[1] - col > patch_size else image.shape[1] - col

    patch_image = image[row: row + patch_image_height, col: col + patch_image_width]

    if patch_image_height < patch_size or patch_image_width < patch_size:
        pad_height = patch_size - patch_image_height
        pad_width = patch_size - patch_image_width
        patch_image = np.pad(patch_image, ((0, pad_height), (0, pad_width), (0, 0)), 'constant')

    return patch_image


def load_geojson(filename):
    """ Gets label data from a geojson label file

    :param (str) filename: file path to a geojson label file
    :return: (numpy.ndarray, numpy.ndarray ,numpy.ndarray) coords, chips, and classes corresponding to
            the coordinates, image names, and class codes for each ground truth.
    """

    with open(filename) as f:
        data = json.load(f)

    obj_coords = np.zeros((len(data['features']), 8))
    image_ids = np.zeros((len(data['features'])), dtype='object')
    class_indices = np.zeros((len(data['features'])), dtype=int)
    class_names = np.zeros((len(data['features'])), dtype='object')

    for idx in range(len(data['features'])):
        properties = data['features'][idx]['properties']
        image_ids[idx] = properties['image_id']
        obj_coords[idx] = np.array([float(num) for num in properties['bounds_imcoords'].split(",")])
        class_indices[idx] = properties['type_id']
        class_names[idx] = properties['type_name']

    return image_ids, obj_coords, class_indices, class_names


def cvt_coords_to_rboxes(coords):
    """ Processes a coordinate array from a geojson into (cy, cx, height, width, theta) format

    :param (numpy.ndarray) coords: an array of shape (N, 8) with 4 corner points of boxes
    :return: (numpy.ndarray) an array of shape (N, 5) with coordinates in proper format
    """

    rboxes = []
    for coord in coords:
        pts = np.reshape(coord, (-1, 2)).astype(dtype=np.float32)
        (cx, cy), (width, height), theta = cv2.minAreaRect(pts)

        if width < height:
            width, height = height, width
            theta += 90.0
        rboxes.append([cy, cx, height, width, math.radians(theta)])

    return np.array(rboxes)


def cvt_coords_to_polys(coords):
    """ Convert a coordinate array from a geojson into Polygons

    :param (numpy.ndarray) coords: an array of shape (N, 8) with 4 corner points of boxes
    :return: (numpy.ndarray) polygons: an array of shapely.geometry.Polygon corresponding to coords
    """

    polygons = []
    for coord in coords:
        polygons.append(Polygon([coord[0:2], coord[2:4], coord[4:6], coord[6:8]]))
    return np.array(polygons)


def IoA(poly1, poly2):
    """ Intersection-over-area (ioa) between two boxes poly1 and poly2 is defined as their intersection area over
    box2's area. Note that ioa is not symmetric, that is, IOA(poly1, poly2) != IOA(poly1, poly2).

    :param (shapely.geometry.Polygon) poly1: Polygon1
    :param (shapely.geometry.Polygon) poly2: Polygon2
    :return: (float) IoA between poly1 and poly2
    """
    return poly1.intersection(poly2).area / poly1.area


def tf_int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def tf_int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def tf_bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def tf_bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def tf_float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def cvt_rbox_to_tfexample(encode_image, image_height, image_width, image_filename, image_format, center_ys, center_xs,
                          heights, widths, thetas, class_texts, class_indices):
    """ Build an Example proto for an example of rbox.

    :param (bytes) encode_image: encoded image
    :param (int) image_height: height of image
    :param (int) image_width: width of image
    :param (bytes) image_filename: encoded image name
    :param (bytes) image_format: encoded file format
    :param (list) center_ys: a list of center y of objects
    :param (list) center_xs: a list of center x of objects
    :param (list) heights: a list of height of objects
    :param (list) widths: a list of width of objects
    :param (list) thetas: a list of theta of objects
    :param (list) class_texts: a list of class text of objects
    :param (list) class_indices: a list of class index of objects
    :return: (tf.train.Example) example proto of rbox
    """

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': tf_int64_feature(image_height),
        'image/width': tf_int64_feature(image_width),
        'image/filename': tf_bytes_feature(image_filename),
        'image/encoded': tf_bytes_feature(encode_image),
        'image/format': tf_bytes_feature(image_format),
        'image/object/rbbox/cx': tf_float_list_feature(center_xs),
        'image/object/rbbox/cy': tf_float_list_feature(center_ys),
        'image/object/rbbox/w': tf_float_list_feature(widths),
        'image/object/rbbox/h': tf_float_list_feature(heights),
        'image/object/rbbox/ang': tf_float_list_feature(thetas),
        'image/object/class/text': tf_bytes_list_feature(class_texts),
        'image/object/class/label': tf_int64_list_feature(class_indices),
    }))
    return example


def write_tfrecords(trf_writer, patches):
    """ Write patch information into writer

       :param (str) dst_tfr_path: path to save tfrecords
       :param (list) patches: a list of Patch to save tfrecords
       :param (str) obj_type: object type which is one of {'rbox', 'bbox'}
    """

    for patch in patches:
        image_format = b'png'
        image = cv2.cvtColor(patch.image, cv2.COLOR_RGB2BGR)
        image_as_bytes = cv2.imencode('.png', image)[1].tostring()

        encoded_image = image_as_bytes
        patch_height = patch.image.shape[0]
        patch_width = patch.image.shape[1]
        image_filename = patch.image_id.encode()

        center_ys, center_xs, heights, widths, thetas, class_indices, class_texts = [], [], [], [], [], [], []
        for coord, cls_idx, cls_text in patch.objects:
            center_ys.append(coord[0] / patch_height)
            center_xs.append(coord[1] / patch_width)
            heights.append(coord[2] / patch_height)
            widths.append(coord[3] / patch_width)
            thetas.append(coord[4])
            class_texts.append(cls_text.encode())
            class_indices.append(cls_idx)

        tfexample = cvt_rbox_to_tfexample(encoded_image, patch_height, patch_width, image_filename, image_format,
                                          center_ys, center_xs, heights, widths, thetas, class_texts, class_indices)

        trf_writer.write(tfexample.SerializeToString())


def create_tfrecords(src_dir, dst_path, patch_size=1024, patch_overlay=384, object_fraction_thresh=0.7,
                     is_include_only_pos=False):
    """ Create TF Records from geojson

    :param (str) src_dir: path to a GeoJson file
    :param (str) dst_path: Path to save tfrecords'
    :param (int) patch_size: patch size
    :param (int) patch_overlay: overlay size for patching
    :param (float) object_fraction_thresh: threshold value for determining contained objects
    :param (bool) is_include_only_pos: Whether or not to include only positive patch image(containing at least one object)
    :return:
    """

    trf_writer = tf.python_io.TFRecordWriter(dst_path)
    n_tfrecord = 0

    # Load objects from geojson
    geojson_path = os.path.join(src_dir, 'labels.json')
    image_ids, obj_coords, class_indices, class_names = load_geojson(geojson_path)

    obj_polys = cvt_coords_to_polys(obj_coords)
    obj_coords = cvt_coords_to_rboxes(obj_coords)

    # Load image files as TIF
    for image_id in tqdm(sorted(set(image_ids))):

        image = imread(os.path.join(src_dir, 'images/', image_id))

        # Get data in the current image
        obj_coords_in_image = obj_coords[image_ids == image_id]
        obj_polys_in_image = obj_polys[image_ids == image_id]
        class_indices_in_image = class_indices[image_ids == image_id]
        class_texts_in_image = class_names[image_ids == image_id]

        # Create patches including objects
        patches = []
        step = patch_size - patch_overlay
        for row in range(0, image.shape[0] - patch_overlay, step):
            for col in range(0, image.shape[1] - patch_overlay, step):
                patch_poly = Polygon([(col, row), (col + patch_size, row),
                                      (col + patch_size, row + patch_size), (col, row + patch_size)])

                # Check if a patch contains objects and append objects
                objects_in_patch = []
                for idx, obj_poly in enumerate(obj_polys_in_image):
                    if IoA(obj_poly, patch_poly) > object_fraction_thresh:
                        objects_in_patch.append(Object(obj_coords_in_image[idx], class_indices_in_image[idx],
                                                       class_texts_in_image[idx]))

                # if a patch contains objects, append the patch to save tfrecords
                if not is_include_only_pos or objects_in_patch:
                    objects_in_patch = [
                        Object(coord=[obj.coord[0] - row, obj.coord[1] - col, obj.coord[2], obj.coord[3], obj.coord[4]],
                               cls_idx=obj.cls_idx, cls_text=obj.cls_text) for obj in objects_in_patch]
                    patch_image = get_patch_image(image, row, col, patch_size)

                    patches.append(
                        Patch(image_id=image_id, image=patch_image, row=row, col=col, objects=objects_in_patch))


        write_tfrecords(trf_writer, patches)
        n_tfrecord += len(patches)

    print('N of TFRecords:', n_tfrecord)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create TF Records from geojson')
    parser.add_argument('--src_dir',
                        type=str,
                        required=True,
                        metavar='DIR',
                        help='Root directory to geojson and images')
    parser.add_argument('--dst_path',
                        type=str,
                        metavar='FILE',
                        default='tfrecords.tfrecords',
                        help='Path to save tfrecords')
    parser.add_argument('--patch_size',
                        type=int,
                        default=768,
                        help='Patch size')
    parser.add_argument('--patch_overlay',
                        type=int,
                        default=256,
                        help='Overlay size for patching')
    parser.add_argument('--object_fraction_thresh',
                        type=float,
                        default=0.7,
                        help='Threshold value for determining contained objects')
    parser.add_argument('--is_include_only_pos',
                        dest='is_include_only_pos',
                        action='store_true',
                        help='Whether or not to include only positive patch image(containing at least one object)')

    args = parser.parse_args()

    create_tfrecords(**vars(args))
