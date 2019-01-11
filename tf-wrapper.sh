#!/bin/bash

export TF_CONFIG=$(bash tf-config.sh)
echo $TF_CONFIG | jq .

exec $@
