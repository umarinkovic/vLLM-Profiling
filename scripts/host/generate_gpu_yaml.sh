#!/usr/bin/env bash

echo "THE SCRIPT generate_gpu_yaml.sh IS CURRENTLY DEFUNCT"
echo "GENERATE THE FILE MANUALLY"
echo "EXAMPLE:"
echo "gpus:
        0:
          name: Radeon RX 7900 XTX 
          device: /dev/dri/renderD128
        1:
          name: AMD Radeon RX 9070 XT  
          device: /dev/dri/renderD129
        2:
          name: AMD Radeon RX 6700 XT          
          device: /dev/dri/renderD130"
# TODO: ENABLE USING ROCMINFO

exit 1

echo "gpus:"

i=0
for dev in /dev/dri/renderD*; do
  name=$(udevadm info --query=property --name="$dev" \
    | grep ID_MODEL= \
    | cut -d= -f2)

  echo "  gpu$i:"
  echo "    name: ${name:-unknown}"
  echo "    device: $dev"

  ((i++))
done