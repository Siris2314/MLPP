#ifndef DNN_HPP
#define DNN_HPP

#include <Eigen/Core>
#include "Config.hpp"
#include "RandomNumGen.hpp"

#include "Layer.hpp"
#include "Layer/FullyConnected.hpp"
#include "Layer/Conv.hpp"
#include "Layer/Pooling.hpp"

#include "output.hpp"
#include "Error/MSE.hpp"
#include "Error/CrossEntropy.hpp"

#include "optimizer.hpp"
#include "OptiAlgos/SGD.hpp"
#include "OptiAlgos/Momentum.hpp"
#include "OptiAlgos/RMSProp.hpp"
#include "OptiAlgos/Adam.hpp"
#include "OptiAlgos/AdaGrad.hpp"
#include "OptiAlgos/AdaDelta.hpp"
#include "OptiAlgos/Nadam.hpp"
#include "OptiAlgos/NAG.hpp"
#include "OptiAlgos/AMSGrad.hpp"
#include "OptiAlgos/AdaMax.hpp"


#endif