#! /bin/bash 

if [[ -z $ML_ROOT ]]; then
    echo "ML_ROOT is not set"
    exit 1
fi

dir=$(echo "$@" | tr a-z A-Z)
model_name_lower=$(echo "$@" | tr A-Z a-z)

mkdir -p $ML_ROOT/$dir/include $ML_ROOT/$dir/src
touch $ML_ROOT/$dir/Makefile
touch $ML_ROOT/$dir/include/"$model_name_lower.hpp"
touch $ML_ROOT/$dir/src/"$model_name_lower.cc"