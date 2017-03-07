#ifndef COCOTTE_MODELS_MODELDISTANCE_H
#define COCOTTE_MODELS_MODELDISTANCE_H


#include <cocotte/distance.h>

namespace Cocotte {
namespace Models {



using ModelDistance = Distance<DistanceType::L_1_XT>;
//using ModelDistance = Distance<DistanceType::L_2_SQUARED_XT>;
//using ModelDistance = Distance<DistanceType::CLOSEST_DIM_X>;



}}



#endif
