#pragma once

#include "phantom/config.hpp"
#include "phantom/lidar.hpp"
#include "phantom/scene.hpp"

namespace phantom {

/// Build scene sampler parameters from the `[scene]` section.
SceneParams scene_params_from_config(const Config& cfg);

/// Build sensor parameters from the `[lidar]`, `[noise]` and `[intensity]` sections.
LidarParams lidar_params_from_config(const Config& cfg);

}  // namespace phantom
