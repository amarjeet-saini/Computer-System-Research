#!/bin/bash


var1=$(date +"%T.%6N")
var2=$(cat /sys/devices/platform/host1x/158c0000.nvdla0/power/runtime_status)
var3=$(cat /sys/devices/gpu.0/load)

while True
do
	echo "${var1}, ${var2}, ${var3}" >> result.txt
done
