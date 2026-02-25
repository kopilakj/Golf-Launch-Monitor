#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
int main() {
    // --- defaults (tune later) ---
    const int width = 320;
    const int height = 240;
    const int fps = 300;
    const int shutter_us = 1000;   // 1 ms
    const double gain = 4.0;
    const int duration_ms = 2000;  // 2 seconds (set 0 for unlimited)
    const std::string out_file = "/dev/null";   // set to "capture.yuv" to save frames
    const std::string pts_file = "capture.pts"; // timestamps
    // --- rpicam-vid command ---
    std::ostringstream cmd;
    cmd << "rpicam-vid"
        << " --width " << width
        << " --height " << height
        << " --framerate " << fps
        << " --codec yuv420"
        << " --shutter " << shutter_us
        << " --gain " << gain
        << " --denoise off"
        << " --nopreview"
        << " --save-pts " << pts_file
        << " -t " << duration_ms
        << " -o " << out_file;
    std::cout << "Running: " << cmd.str() << std::endl;
    return std::system(cmd.str().c_str());
}