/// Copyright (c)2021 Electronics and Telecommunications Research Institute(ETRI)

#include "deepnet_test/DeepNetTest.hpp"
#include "deepnet/Network.hpp"
#include "deepnet/dataset/DatasetImageNet.hpp"
#include "deepnet/Image.hpp"
#include "deepnet/Timer.hpp"
#include <opencv2/opencv.hpp>


using namespace deepnet;

DEEPNET_TEST_BEGIN(TestDatasetImageNet, !deepnet_test::autorun)
{
    auto imagenet_index_path = "../data/image/ImageNet/ILSVRC2012/ILSVRC2012_img_train/index.txt";

    auto dataset = DatasetImageNet(imagenet_index_path, 5000);

    auto batch_size = 5;

    TensorCpu x(batch_size, 3, 224, 224);
    TensorCpu y(batch_size, 1000, 1, 1);

    dataset.shuffle();

    Timer timer;
    auto success = dataset.get(x, y);
    DEEPNET_ASSERT(success);
    DEEPNET_LOG("Elapsed time = " << timer.elapsed() << "ms.");

    for (auto i = 0; i < batch_size; i++)
        y.print("y = ");

    x.show("image");
    cv::waitKey();
}
DEEPNET_TEST_END(TestDatasetImageNet)
