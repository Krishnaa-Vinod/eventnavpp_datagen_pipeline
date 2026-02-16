#ifndef EVENT_LANDMARK_EXTRACTOR_HPP
#define EVENT_LANDMARK_EXTRACTOR_HPP

// #include "rclcpp/rclcpp.hpp"
// #include "sensor_msgs/msg/image.hpp"
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>


class EventLandmarkExtractor {
    public:
    void event_feature_extraction(const sensor_msgs::msg::Image voxel_grid);

};

#endif