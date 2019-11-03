#include "PlarformEx.h"

std::vector<cl::Platform> PlarformEx::GetPlatforms()
{
	auto platforms = std::vector<cl::Platform>();
	cl::Platform::get(&platforms);
	return platforms;
}

std::vector<cl::Device> PlarformEx::GetDevices(const cl::Platform& platform)
{
	auto devices = std::vector<cl::Device>();
	platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
	return devices;
}