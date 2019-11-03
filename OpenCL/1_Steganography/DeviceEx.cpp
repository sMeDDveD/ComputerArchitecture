#include "DeviceEx.h"
#include "PlarformEx.h"

std::string DeviceEx::GetVendor(const cl::Device& device)
{
	return std::string(device.getInfo<CL_DEVICE_VENDOR>());
}

std::string DeviceEx::GetName(const cl::Device& device)
{
	return std::string(device.getInfo<CL_DEVICE_NAME>());
}

cl::Device DeviceEx::Find(std::string name)
{
	for (auto platform : PlarformEx::GetPlatforms())
	{
		for (auto device : PlarformEx::GetDevices(platform))
		{
			auto deviceName = GetName(device);
			auto vendor = GetVendor(device);
			if (deviceName.find(name) != std::string::npos || vendor.find(name) != std::string::npos)
			{
				return device;
			}
		}
	}

	throw std::exception("Can't find OpenCL device!");
}

cl::Context DeviceEx::CreateContext(const cl::Device& device)
{
	auto contextDevices = std::vector<cl::Device>();
	contextDevices.push_back(device);
	return cl::Context(contextDevices);
}