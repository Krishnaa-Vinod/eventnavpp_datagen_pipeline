#ifndef VIS_NODE_HPP
#define VIS_NODE_HPP


#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "event_camera_msgs/msg/event_packet.hpp"
#include "Eigen/Dense"
// #include "Eigen/Tensor"
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <event_camera_codecs/decoder.h>
#include <event_camera_codecs/decoder_factory.h>

struct Event {
    uint16_t x;
    uint16_t y;
    uint64_t t;   // timestamp in nanoseconds
    bool polarity; // true = ON, false = OFF
};


class NavBagVisNode : public rclcpp::Node, public event_camera_codecs::EventProcessor {
    public:
        NavBagVisNode();

        void eventCD(uint64_t sensor_time, uint16_t ex, uint16_t ey, uint8_t polarity) override;
        bool eventExtTrigger(uint64_t sensor_time, uint8_t edge, uint8_t id) override {}
        void finished() override {}
        void rawData(const char*, size_t) override {}
    private:
        rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr rgb_cam_sub_;
        rclcpp::Subscription<event_camera_msgs::msg::EventPacket>::SharedPtr event_raw_sub_;

        rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr event_voxel_pub_;
        rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr rgb_flip_frame_pub_;
        // rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr event_frame_pub_;

        int height_;
        int width_;
        int num_bins_;
        double accumulation_time_;
        std::array<int, 3> voxel_size_;

        std::vector<Event> event_buffer_;
        event_camera_codecs::DecoderFactory<event_camera_msgs::msg::EventPacket, NavBagVisNode> decoder_factory_;


        void eventCallback(const event_camera_msgs::msg::EventPacket::SharedPtr msg);
        void rgbCallback(const sensor_msgs::msg::Image::SharedPtr msg);

        void publishVoxelGrid(const std::vector<Event> &event_buffer_, const std_msgs::msg::Header &header);
        // void publishVoxelGrid(const std::vector<Event> &event_buffer_, const std_msgs::msg::Header &header)
        // rclcpp::Subcription<>

};


#endif


