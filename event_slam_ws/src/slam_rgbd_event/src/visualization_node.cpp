#include <iostream>
#include "rclcpp/rclcpp.hpp"
#include "slam_rgbd_event/visualization_node.hpp"


NavBagVisNode::NavBagVisNode() : Node("vis_node") {

    rclcpp::QoS qos(10);
    qos.best_effort();
    qos.durability_volatile();

    event_raw_sub_ = this->create_subscription<event_camera_msgs::msg::EventPacket>(
        "/event_camera/events",
        qos, 
        std::bind(&NavBagVisNode::eventCallback, this, std::placeholders::_1)
    );

    rgb_cam_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
        "/cam_sync/cam0/image_raw",
        10, 
        std::bind(&NavBagVisNode::rgbCallback, this, std::placeholders::_1)
    );

    event_voxel_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
                        "/event_camera/voxel_grid",
                            10 
                    );

    rgb_flip_frame_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
                        "/rgb_camera/image",
                            10 
                    );

    height_ = 720;
    width_ = 1280;
    num_bins_ = 5;
    voxel_size_ = {height_, width_, num_bins_};
    // resoluation of the evk3 event camera. 


}


void NavBagVisNode::eventCD(uint64_t sensor_time, uint16_t ex, uint16_t ey, uint8_t polarity) {
    Event e;
    e.x = ex;
    e.y = ey;
    e.t = sensor_time;
    e.polarity = (polarity != 0);
    event_buffer_.push_back(e);
}

void NavBagVisNode::eventCallback(const event_camera_msgs::msg::EventPacket::SharedPtr msg) {
    event_buffer_.clear();
    auto decoder = decoder_factory_.getInstance(*msg);

    if (!decoder) { // msg->encoding was invalid
            return;
        }
    decoder->decode(*msg, this);

    if (event_buffer_.empty()) return;

    RCLCPP_INFO_ONCE(this->get_logger(),
        "Decoded %zu events from first %s packet", event_buffer_.size(), msg->encoding.c_str());
    
    const uint64_t window_us = 50000;
    uint64_t t_start = std::numeric_limits<uint64_t>::max();
    uint64_t t_end = 0;
    for (const auto& e : event_buffer_) {
        if (e.t < t_start) t_start = e.t;
        if (e.t > t_end) t_end = e.t;
    }

    size_t idx = 0;
    while (t_start < t_end) {
        uint64_t t_window_end = t_start + window_us;
        std::vector<Event> window_events;

        // Collect events in the current window
        while (idx < event_buffer_.size() && event_buffer_[idx].t < t_window_end) {
            window_events.push_back(event_buffer_[idx]);
            ++idx;
        }

        if (!window_events.empty()) {
            publishVoxelGrid(window_events, msg->header);
        }

        t_start = t_window_end;
    }


}


void NavBagVisNode::publishVoxelGrid(const std::vector<Event> &event_buffer_, const std_msgs::msg::Header &header) {
    // std::cout << event_buffer_;
    RCLCPP_INFO_ONCE(this->get_logger(),
        "First event in EventPacket: ( %d, %d, %d, %lu)", event_buffer_[0].x, event_buffer_[0].y, event_buffer_[0].polarity, event_buffer_[0].t);
        
    
    cv::Mat voxel_grid = cv::Mat::zeros(3, voxel_size_.data(), CV_32FC1);

    uint64_t t_min = event_buffer_[0].t;
    uint64_t t_max = event_buffer_[0].t;
    for (const auto &event : event_buffer_) {
        if (event.t < t_min) t_min = event.t;
        if (event.t > t_max) t_max = event.t;
    }

    double dt = static_cast<double>(t_max - t_min);
    RCLCPP_INFO_ONCE(this->get_logger(),
        "dt ( %f)", dt);

    if(dt <= 0) return;

    // uint64_t t_min = event_buffer_.front().t;
    for (const auto &event : event_buffer_) {
        double bin_norm = (static_cast<double>(event.t - t_min ) / dt ) * (num_bins_- 1);
        int bin_low = floor(bin_norm);
        int bin_high = bin_low + 1;

        double w_high = static_cast<double>(bin_norm - bin_low);
        double w_low = 1.0 - w_high;
        float pol = event.polarity  ? 1.0f : -1.0f;
        
        voxel_grid.at<float>(event.y, event.x, bin_low) += pol * w_low ;

        if (bin_high < num_bins_) {
            voxel_grid.at<float>(event.y, event.x, bin_high) += pol * w_high ;
        }
        
    }

    cv::Mat voxel_image(height_, width_, CV_32FC(num_bins_), voxel_grid.data);

    auto msg = cv_bridge::CvImage(header, "32FC5", voxel_image).toImageMsg();
    event_voxel_pub_->publish(*msg);

}

void NavBagVisNode::rgbCallback(const sensor_msgs::msg::Image::SharedPtr msg) {
    // RCLCPP_INFO(this->get_logger(), "rgb callback");
    cv_bridge::CvImagePtr cv_ptr;
    
    cv_ptr = cv_bridge::toCvCopy(msg, "bgr8");
    cv::flip(cv_ptr->image, cv_ptr->image, 1);

    auto flipped_msg = cv_ptr->toImageMsg();
    flipped_msg->header = msg->header;
    rgb_flip_frame_pub_->publish(*flipped_msg);
    
}

int main(int argc, char * argv[]) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<NavBagVisNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
