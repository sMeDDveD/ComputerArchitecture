#include "KernelEx.h"
#include <fstream>

std::string LoadText(const std::string& path)
{
	auto sourceFile = std::ifstream(path.c_str());
	return std::string(std::istreambuf_iterator<char>(sourceFile), std::istreambuf_iterator<char>());
}

cl::Kernel KernelEx::BuildFromFile(const cl::Device& device, const cl::Context& context, const std::string& kernelFile, const std::string& kernelName)
{
	auto kernelSource = LoadText(kernelFile);
	return BuildFromSource(device, context, kernelSource, kernelName);
}

cl::Kernel KernelEx::BuildFromSource(const cl::Device& device, const cl::Context& context, const std::string& kernelCode, const std::string& kernelName)
{
	auto source = cl::Program::Sources(1, std::make_pair(kernelCode.c_str(), kernelCode.length() + 1));
	cl_int error;
	auto program = cl::Program(context, source, &error);
	auto contextDevices = std::vector<cl::Device>{ device };
	program.build(contextDevices);
	return cl::Kernel(program, kernelName.c_str());
}