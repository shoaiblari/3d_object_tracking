import json

import cv2
import numpy as np
from pyquaternion import Quaternion

from nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import BoxVisibility
from nuscenes.utils.geometry_utils import box_in_image

file_url = '/home/abhinav/PycharmProjects/mahalanobis_3d_multi_object_tracking/results/000008/train' \
           '/results_train_probabilistic_tracking.json'
data_root = '/home/abhinav/PycharmProjects/mahalanobis_3d_multi_object_tracking/dataset/nuscenes/trainval/v1.0-mini'
version = 'v1.0-mini'

nusc = NuScenes(version=version, dataroot=data_root, verbose=True)

with open(file_url, 'r') as f:
    tracking_results = json.load(f)['results']

# For each sample of a scene, look up in results to find the
# estimate co-ordinates of the objects and plot it

available_scenes = nusc.scene
selected_scene = available_scenes[0]

first_sample_token = selected_scene['first_sample_token']
curr_sample_token = first_sample_token

annotations_to_track = ['car', 'bus']

img_size = 100
visibility_level = BoxVisibility.ANY

Hz = 10
video = None

while curr_sample_token != '':
    curr_sample = nusc.get('sample', curr_sample_token)
    tracked_annotations = tracking_results[curr_sample_token]

    # High level algorithm
    # NOTE: annotation and box is used interchangeably
    # We need to move the box from global co-ordinates to front camera co-ordinates
    # We have the following transformations
    # 1) globalTego_pose : car pose in global co-ordinates
    # 2) ego_poseTcam_front : front camera in car co-ordinates
    #
    # To move an object from global to cam_front we need the following
    # transformation
    # cam_frontTglobal =  (ego_poseTcam_front)^-1 (globalT^ego_pose)^-1

    front_cam_data_token = curr_sample['data']['CAM_FRONT']
    front_cam_data = nusc.get('sample_data', front_cam_data_token)

    ego_pose_data = nusc.get('ego_pose', front_cam_data['ego_pose_token'])

    # https: // www.nuscenes.org / tutorials / nuscenes_tutorial.html
    # Refer 10
    cam_front_sensor_data = nusc.get('calibrated_sensor',
                                     front_cam_data['calibrated_sensor_token'])  # ego_poseTcam_front

    ego_pose_record = nusc.get('ego_pose', front_cam_data['ego_pose_token'])  # globalTego_pose

    # Find the camera intrinsic
    K = np.array(cam_front_sensor_data['camera_intrinsic'])

    img_size = (front_cam_data['width'], front_cam_data['height'])

    # Find the image for the sample
    image_url = nusc.get_sample_data_path(front_cam_data_token)

    # Get the visible annotations in cam front co-ordinates
    visible_boxes = []
    for annotations_in_sample in tracked_annotations:
        if annotations_in_sample['tracking_name'] in annotations_to_track:

            # The box is in global co-ordinates
            box = Box(annotations_in_sample['translation'], annotations_in_sample['size'],
                      Quaternion(annotations_in_sample['rotation']))

            # Move the box from global co-ordinates to ego pose co-ordinates
            box.translate(-np.array(ego_pose_record['translation']))
            box.rotate(Quaternion(ego_pose_record['rotation']).inverse)

            # Move the box from ego pose-coordinates to front camera co-ordinates
            box.translate(-np.array(cam_front_sensor_data['translation']))
            box.rotate(Quaternion(cam_front_sensor_data['rotation']).inverse)

            # check if the box is visible in the 2D image(Use camera intrinsics)
            if box_in_image(box, K, img_size, visibility_level):
                visible_boxes.append(box)

    # end of visible annotation computation

    image = cv2.imread(image_url)

    for box in visible_boxes:
        box.render_cv2(image, K, normalize=True)

    cv2.imshow('image', image)
    cv2.waitKey(100)

    height, width, layers = image.shape
    if video is None:
        video = cv2.VideoWriter('kalman_output.avi', 0, Hz, (width, height))

    video.write(image)

    curr_sample_token = curr_sample['next']

cv2.destroyAllWindows()
video.release()