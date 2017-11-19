#!/usr/bin/env python

# Import modules
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
import pickle
from sensor_stick.srv import GetNormals
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from visualization_msgs.msg import Marker
from sensor_stick.marker_tools import *
from sensor_stick.msg import DetectedObjectsArray
from sensor_stick.msg import DetectedObject
from sensor_stick.pcl_helper import *

import rospy
import tf
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
from std_msgs.msg import Int32
from std_msgs.msg import String
from pr2_robot.srv import *
from rospy_message_converter import message_converter
import yaml

from std_msgs.msg import Int32
from std_msgs.msg import String
from geometry_msgs.msg import Pose


# Helper function to get surface normals
def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy("/feature_extractor/get_normals", GetNormals)
    return get_normals_prox(cloud).cluster

# Helper function to create a yaml friendly dictionary from ROS messages
def make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose):
    yaml_dict = {}
    yaml_dict["test_scene_num"] = test_scene_num.data
    yaml_dict["arm_name"]  = arm_name.data
    yaml_dict["object_name"] = object_name.data
    yaml_dict["pick_pose"] = message_converter.convert_ros_message_to_dictionary(pick_pose)
    yaml_dict["place_pose"] = message_converter.convert_ros_message_to_dictionary(place_pose)
    return yaml_dict

# Helper function to output to yaml file
def send_to_yaml(yaml_filename, dict_list):
    data_dict = {"object_list": dict_list}
    with open(yaml_filename, "w") as outfile:
        yaml.dump(data_dict, outfile, default_flow_style=False)

# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):

    # Convert ROS msg to PCL data
    cloud = ros_to_pcl(pcl_msg)
    
    # Statistical Outlier Filtering
    MEAN_K = 50
    THRESHOLD = 1.0
    outlier_filter = cloud.make_statistical_outlier_filter()
    outlier_filter.set_mean_k(MEAN_K)
    outlier_filter.set_std_dev_mul_thresh(THRESHOLD)
    cloud_filtered = outlier_filter.filter()

    # Voxel Grid Downsampling
    LEAF_SIZE = 0.01
    vox_filter = cloud_filtered.make_voxel_grid_filter()
    vox_filter.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)
    cloud_filtered = vox_filter.filter()

    # PassThrough Filter Z
    AXIS_MIN = 0.6
    AXIS_MAX = 1.1
    FILTER_AXIS = "z"
    passthrough = cloud_filtered.make_passthrough_filter()
    passthrough.set_filter_field_name(FILTER_AXIS)
    passthrough.set_filter_limits(AXIS_MIN, AXIS_MAX)
    cloud_filtered = passthrough.filter()

    # PassThrough Filter x
    AXIS_MIN = -0.5
    AXIS_MAX = 0.5
    FILTER_AXIS = "y"
    passthrough = cloud_filtered.make_passthrough_filter()
    passthrough.set_filter_field_name(FILTER_AXIS)
    passthrough.set_filter_limits(AXIS_MIN, AXIS_MAX)
    cloud_filtered = passthrough.filter()

    # RANSAC Plane Segmentation
    MAX_DISTANCE = 0.01
    seg = cloud_filtered.make_segmenter()
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)
    seg.set_distance_threshold(MAX_DISTANCE)
    inliers, coefficients = seg.segment()

    # Extract inliers and outliers
    cloud_table = cloud_filtered.extract(inliers, negative=False)
    cloud_objects = cloud_filtered.extract(inliers, negative=True)

    # Create a cluster extraction object
    white_cloud = XYZRGB_to_XYZ(cloud_objects)
    tree = white_cloud.make_kdtree()
    ec = white_cloud.make_EuclideanClusterExtraction()
    
    # Set tolerances for distance threshold
    TOLERANCE = 0.02
    MIN_CLUSTER_SIZE = 10
    MAX_CLUSTER_SIZE = 1000
    ec.set_ClusterTolerance(TOLERANCE)
    ec.set_MinClusterSize(MIN_CLUSTER_SIZE)
    ec.set_MaxClusterSize(MAX_CLUSTER_SIZE)
    # Search the k-d tree for clusters
    ec.set_SearchMethod(tree)
    # Extract indices for each of the discovered clusters
    cluster_indices = ec.Extract()

    # Create Cluster-Mask Point Cloud to visualize each cluster separately
    #Assign a color corresponding to each segmented object in scene
    cluster_color = get_color_list(len(cluster_indices))

    color_cluster_point_list = []

    for j, indices in enumerate(cluster_indices):
        for i, indice in enumerate(indices):
            color_cluster_point_list.append([white_cloud[indice][0],
                                            white_cloud[indice][1],
                                            white_cloud[indice][2],
                                             rgb_to_float(cluster_color[j])])

    #Create new cloud containing all clusters, each with unique color
    cluster_cloud = pcl.PointCloud_PointXYZRGB()
    cluster_cloud.from_list(color_cluster_point_list)

    # Convert PCL data to ROS messages
    ros_cloud_table = pcl_to_ros(cloud_table)
    ros_cloud_objects = pcl_to_ros(cloud_objects)
    ros_cluster_cloud = pcl_to_ros(cluster_cloud)

    # Publish ROS messages
    pcl_table_pub.publish(ros_cloud_table)
    pcl_objects_pub.publish(ros_cloud_objects)
    pcl_cluster_pub.publish(ros_cluster_cloud)

    # Classify the clusters! (loop through each detected cluster one at a time)
    detected_objects_labels = []
    detected_objects = []

    for index, pts_list in enumerate(cluster_indices):
        # Grab the points for the cluster from the extracted outliers (cloud_objects)
        pcl_cluster = cloud_objects.extract(pts_list)
        # convert the cluster from pcl to ROS using helper function
        cloud = pcl_to_ros(pcl_cluster)

        # Extract histogram features
        chists = compute_color_histograms(cloud, using_hsv=True)
        normals = get_normals(cloud)
        nhists = compute_normal_histograms(normals)
        feature = np.concatenate((chists, nhists))

        # Make the prediction, retrieve the label for the result
        # and add it to detected_objects_labels list
        prediction = clf.predict(scaler.transform(feature.reshape(1,-1)))
        label = encoder.inverse_transform(prediction)[0]
        detected_objects_labels.append(label)

        # Publish a label into RViz
        label_pos = list(white_cloud[pts_list[0]])
        label_pos[2] += .4
        object_markers_pub.publish(make_label(label,label_pos, index))

        # Add the detected object to the list of detected objects.
        do = DetectedObject()
        do.label = label
        do.cloud = ros_cluster_cloud
        detected_objects.append(do)

    rospy.loginfo("Detected {} objects: {}".format(len(detected_objects_labels), detected_objects_labels))

    # Publish the list of detected objects
    # This is the output you'll need to complete the upcoming project!
    detected_objects_pub.publish(detected_objects)

    # Suggested location for where to invoke your pr2_mover() function within pcl_callback()
    # Could add some logic to determine whether or not your object detections are robust
    # before calling pr2_mover()
    try:
        pr2_mover(detected_objects)
    except rospy.ROSInterruptException:
        pass

# function to load parameters and request PickPlace service
def pr2_mover(object_list):

    # Initialize variables
    test_scene_num = Int32()
    object_name = String()
    arm_name = String()
    pick_pose = Pose()
    place_pose = Pose()
    
    dict_list = []
    labels = []
    centroids = []

    scene = 3
    
    # Loop through detected objects list and create centroids
    for object in object_list:
        labels.append(object.label)
        points_arr = ros_to_pcl(object.cloud).to_array()
        centroids.append(np.mean(points_arr, axis=0)[:3])
    
    
    # Get/Read parameters
    object_list_param = rospy.get_param("/object_list")
    dropbox_param = rospy.get_param("/dropbox")
    dropox_left = dropbox_param[0]
    dropox_right = dropbox_param[1]
    
    # TODO: Rotate PR2 in place to capture side tables for the collision map

    # Loop through the pick list
    for i in range(len(object_list_param)):
        # Parse parameters into individual variables
        label = object_list_param[i]["name"]
        group = object_list_param[i]["group"]

        # Get the PointCloud for a given object and obtain it's centroid
        try:
            index = labels.index(label)
            centroid = centroids[index]
            
            # Create 'place_pose' for the object
            # Assign the arm to be used for pick_place
            # Create a list of dictionaries (made with make_yaml_dict()) for later output to yaml format
            test_scene_num.data = scene
            object_name.data = object_list_param[i]["name"]
            arm_name.data = "right" if group == "green" else "left" 
            pick_pose.position.x = np.asscalar(centroid[0])
            pick_pose.position.y = np.asscalar(centroid[1])
            pick_pose.position.z = np.asscalar(centroid[2])
            place_pose.position.x = dropox_right["position"][0] if group == "green" else dropox_left["position"][0]
            place_pose.position.y = dropox_right["position"][1] if group == "green" else dropox_left["position"][1]
            place_pose.position.z = dropox_right["position"][2] if group == "green" else dropox_left["position"][2]
            
            yaml_dict = make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose)
            dict_list.append(yaml_dict)
        except (ValueError,IndexError):
            print "Object recognition failed: Expected " + label + " and predicted "  + object_list[i].label

        # Wait for 'pick_place_routine' service to come up
        rospy.wait_for_service("pick_place_routine")

        #try:
        #    pick_place_routine = rospy.ServiceProxy("pick_place_routine", PickPlace)

            # Insert your message variables to be sent as a service request
        #    resp = pick_place_routine(test_scene_num, object_name, arm_name, pick_pose, place_pose)

        #    print ("Response: ",resp.success)

        #except rospy.ServiceException, e:
        #    print "Service call failed: %s"%e

    # Output your request parameters into output yaml file
    file = "../config/output_" + str(scene) + ".yml"
    send_to_yaml(file, dict_list)


if __name__ == '__main__':

    # ROS node initialization
    rospy.init_node("perception", anonymous=True)

    # Create Subscribers
    pcl_sub = rospy.Subscriber("pr2/world/points", pc2.PointCloud2, pcl_callback, queue_size=1)

    # Create Publishers
    pcl_objects_pub = rospy.Publisher("/pcl_objects", PointCloud2, queue_size=1)
    pcl_table_pub = rospy.Publisher("/pcl_table", PointCloud2, queue_size=1)
    pcl_cluster_pub = rospy.Publisher("/pcl_cluster", PointCloud2, queue_size=1)
    object_markers_pub = rospy.Publisher("/object_markers", Marker, queue_size=1)
    detected_objects_pub = rospy.Publisher("/detected_objects", DetectedObjectsArray, queue_size=1)

    # TODO: Load Model From disk
    model = pickle.load(open("model.sav", "rb"))
    clf = model["classifier"]
    encoder = LabelEncoder()
    encoder.classes_ = model["classes"]
    scaler = model["scaler"]

    # Initialize lists
    get_color_list.color_list = []
    detected_objects_list = []

    # Spin while node is not shutdown
    while not rospy.is_shutdown():
        rospy.spin()