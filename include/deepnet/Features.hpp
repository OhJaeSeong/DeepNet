#pragma once

// The feature whether we use OPENCV
// If we use OPENCV, all the operations with related to TensorCpu is prohibited.
// Therefore, we should use GPU operations only...
/// opencv 사용여부를 선택하는 (1 = 사용, 0 = 미사용) 옵션
#define FEATURE_USE_OPENCV  1