#!/usr/bin/env python3
import rospy
import numpy as np
from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge
import std_msgs.msg
import sensor_msgs.point_cloud2 as pc2
import cv2
from message_filters import ApproximateTimeSynchronizer, Subscriber
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import PointCloud
# Initialize global variables
rgbimage = None
bridge = CvBridge()
pdata = []
# Camera intrinsics for RGB
camera_intrinsicsrgb = np.array([
    [343.159, 0, 320.5],
    [0, 343.159, 240.5],
    [0, 0, 1]
])

# Initialize the publisher
pub = None

# Define the callback function for synchronized depth and RGB images
def depth_callback(dimage, rgbImage):
    global rgbimage, pub, check
    try:
        # Convert ROS image messages to OpenCV images
        depth_image = bridge.imgmsg_to_cv2(dimage, desired_encoding="passthrough")
        rgbimage = bridge.imgmsg_to_cv2(rgbImage, desired_encoding="bgr8")  # Convert RGB to BGR format

        # Call depth to point cloud conversion
        depth_to_point_cloud(depth_image, rgbimage)

    except Exception as e:
        rospy.logerr(f"Error in depth_callback: {e}")

def depth_to_point_cloud(depth_image, rgbimage):
    global check
    goAhead = False
    objFound = False
    height, width = depth_image.shape
    u_coords, v_coords = np.meshgrid(np.arange(0, height, 20), np.arange(0, width, 20))

    x = (v_coords - camera_intrinsicsrgb[0, 2]) * depth_image[u_coords, v_coords] / camera_intrinsicsrgb[0, 0]
    y = (u_coords - camera_intrinsicsrgb[1, 2]) * depth_image[u_coords, v_coords] / camera_intrinsicsrgb[1, 1]
    z = depth_image[u_coords, v_coords]
    y, z, x = -x, -y, z

    mask = (-0.5 < z)
    x, y, z = x[mask], y[mask], z[mask]
    mask1 = (x < 6) & (x > 0.5)
    x, y, z = x[mask1], y[mask1], z[mask1]
    #points = np.stack((x, y, z), axis=-1).astype(np.float32) # only point cloud
    ppoints = []
    shifted_points = []
    pointK = []
    
    if rgbimage is not None:
        ppoints = add_points(rgbimage, depth_image)

        if ppoints is not None and ppoints.size > 0:
            shifted_points = ppoints + np.array([0, 0, 0.2])  # Add [0, 0, 0.2] to each point
            pointsR = shifted_points + np.array([0, 0, 0.2])
            goAhead = True
    
    if pdata is not None and hasattr(pdata, 'points') and len(pdata.points) > 0:
        objFound = True
        print("Welcome")
        # Extract points from pdata (assuming it is a PointCloud message)
        pointK = np.array([[point.x, point.y, point.z] for point in pdata.points])

        # Apply transformations to the points (example)
        rdata = pointK + np.array([0, 0, 0.4])  # Offset each point by [0, 0, 0.4]
        kdata = rdata + np.array([0, 0, 0.2])  # Further offset each point by [0, 0, 0.2]
        pdata.points = []

    if len(x) == 0:
        rospy.logwarn("No valid points to publish.")
        return

    points = np.stack((x, y, z), axis=-1).astype(np.float32)
    if ppoints is not None and goAhead:
        points = np.vstack((points, ppoints, shifted_points, pointsR))
    
    if objFound:
        points = np.vstack((points, rdata, kdata))
    
    points = points.reshape(-1, 3)

    if points.shape[0] == 0:
        rospy.logwarn("No valid points to publish.")
        return

    header = std_msgs.msg.Header()
    header.stamp = rospy.Time.now()
    header.frame_id = 'camera_frame'

    try:
        # Create PointCloud2 message
        point_cloud_msg = create_cloud_with_intensity(header, points)
        pub.publish(point_cloud_msg)
        #rospy.loginfo("Point cloud published successfully.")
    except Exception as e:
        rospy.logerr(f"Error creating PointCloud2 message: {e}")

def create_cloud_with_intensity(header, points, intensity_value=0.5):
    """
    Create a PointCloud2 message with XYZI fields properly serialized.

    The local planner (localPlanner.cpp) reads point.intensity and only registers
    an obstacle when intensity > obstacleHeightThre (default 0.12).  The previous
    implementation appended the 'intensity' field descriptor to the message header
    but still called create_cloud_xyz32, which only writes XYZ bytes – the intensity
    channel was never actually written into the binary data buffer, so every depth
    point had intensity=0 and was silently ignored by the local planner, causing
    the robot to collide with obstacles.

    Args:
        header         : std_msgs.msg.Header
        points         : Nx3 numpy float32 array (x, y, z)
        intensity_value: float written into the intensity channel for every point.
                         Must be > obstacleHeightThre (0.12).  Default 0.5.
    """
    # Define all four XYZI fields explicitly so create_cloud packs them correctly.
    fields = [
        pc2.PointField(name='x',         offset=0,  datatype=pc2.PointField.FLOAT32, count=1),
        pc2.PointField(name='y',         offset=4,  datatype=pc2.PointField.FLOAT32, count=1),
        pc2.PointField(name='z',         offset=8,  datatype=pc2.PointField.FLOAT32, count=1),
        pc2.PointField(name='intensity', offset=12, datatype=pc2.PointField.FLOAT32, count=1),
    ]

    # Build Nx4 array: [x, y, z, intensity]
    intensity_col = np.full((points.shape[0], 1), intensity_value, dtype=np.float32)
    points_xyzi   = np.hstack((points.astype(np.float32), intensity_col))

    # create_cloud serialises every field – intensity bytes are now in the blob.
    point_cloud_msg = pc2.create_cloud(header, fields, points_xyzi)
    return point_cloud_msg

def add_points(rgimage, gimage):
    hsv_image = cv2.cvtColor(rgimage, cv2.COLOR_BGR2HSV)
    lower_red_1 = np.array([0, 120, 70])
    upper_red_1 = np.array([10, 255, 255])
    lower_red_2 = np.array([170, 120, 70])
    upper_red_2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv_image, lower_red_1, upper_red_1)
    mask2 = cv2.inRange(hsv_image, lower_red_2, upper_red_2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    kernel = np.ones((3, 3), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

    coordinates = np.column_stack(np.where(red_mask > 0))
    coordinates = coordinates[::20]

    if coordinates.size > 0 and np.sum(red_mask) > 100:
        depth_values = gimage[coordinates[:, 0], coordinates[:, 1]]
        valid_mask = depth_values > 0
        coordinates = coordinates[valid_mask]
        depth_values = depth_values[valid_mask]

        x = (coordinates[:, 1] - camera_intrinsicsrgb[0, 2]) * depth_values / camera_intrinsicsrgb[0, 0]
        y = (coordinates[:, 0] - camera_intrinsicsrgb[1, 2]) * depth_values / camera_intrinsicsrgb[1, 1]
        z = depth_values

        y, z, x = -x, -y, z
        y = y + 0.059
        maskr = (x < 6) & (x > 0.4)
        x, y, z = x[maskr], y[maskr], z[maskr]
        tpoints = np.stack((x, y, z), axis=-1).astype(np.float32)
        return tpoints
    else:
        #rospy.loginfo("No red color detected.")
        return None
def yolo_callback(ydata):
    global pdata
    try:
        rospy.loginfo(f"Cup Found!")
        pdata = ydata
        print(f"pdata type: {type(pdata)}")
        rospy.loginfo(f"Number of points in pdata: {len(pdata.points)}")
    except Exception as e:
        rospy.logerr(f"Error in yolo_callback: {e}")
    

def main():
    global pub
    rospy.init_node('Node_Pcloud', anonymous=False)
    pub = rospy.Publisher('/velodyne_points', PointCloud2, queue_size=1)
    rospy.Subscriber('/way_pointp',PointCloud,yolo_callback)
    depth_sub = Subscriber('/camera/depth/image_rect_raw', Image)
    rgb_sub = Subscriber('/camera/image', Image)

    ats = ApproximateTimeSynchronizer([depth_sub, rgb_sub], queue_size=10, slop=0.1)
    ats.registerCallback(depth_callback)

    rospy.spin()

if __name__ == '__main__':
    main()
