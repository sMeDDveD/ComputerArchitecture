#include "OpenCL.h"
#include "DeviceEx.h"
#include "KernelEx.h"

#include <fstream>
#include <iostream>
#include <cstddef>
#include <filesystem>

class Main
{
public:
	using byte = char;

	static void Run(int argc, char* argv[])
	{
		if (argc > 1) 
		{
			try 
			{
				InitializeFromBin(argv[1]);
				auto device = DeviceEx::Find("NVIDIA");
				if (strcmp(argv[2], "-e") == 0 && argc == 5) 
				{
					std::cout << "Encrypting..." << std::endl;
					Encrypting(argv[4], device, argv[5]);
				}
				else if(strcmp(argv[2], "-d") == 0 && argc == 3)
				{
					std::cout << "Decrypting..." << std::endl;
					Decrypting(device);
				}
				else if (strcmp(argv[2], "-t") == 0)
				{
					std::cout << "Testing..." << std::endl;
					Encrypting("test.bin", device, argv[3]);
					InitializeFromBin("test.bin");
					Decrypting(device);
				}
				else
				{
					std::cout << "Unsupported args!" << std::endl;
				}
			}
			catch (const std::exception &e)
			{
				std::cerr << e.what() << std::endl;
			}
		} 
		else
		{
			std::cout << "Enter the command line args!" << std::endl;
		}
	}

private:
	static int n, m;
	static int numOfBytes;
	static byte* bytes;

	template <class T>
	static void endswap(T *objp)
	{
		byte *memp = reinterpret_cast<byte*>(objp);
		std::reverse(memp, memp + sizeof(T));
	}

	static void Encrypting(const char* outputFile, const cl::Device& device, byte* toEncrypt) 
	{
		auto sizeOfStr = strlen(toEncrypt) + 1;
		if (numOfBytes / 8 < sizeOfStr) 
		{
			throw new std::exception("The file is too small!");
		}

		auto context = DeviceEx::CreateContext(device);
		auto bytesBuffer = CreateBuffer(context, bytes, numOfBytes, CL_MEM_READ_WRITE);
		auto toEncryptBuffer = CreateBuffer(context, toEncrypt, sizeOfStr, CL_MEM_READ_ONLY);

		auto kernel = KernelEx::BuildFromFile(device, context, "Kernel.cl", "Encrypt");
		kernel.setArg(0, bytesBuffer);
		kernel.setArg(1, toEncryptBuffer);

		auto queue = cl::CommandQueue(context, device);
		queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(sizeOfStr));
		queue.finish();

		ReadFromBuffer(queue, bytes, numOfBytes, bytesBuffer);

		PrintEncryptedToBin(outputFile);
		DestroyResources();
	}

	static void Decrypting(const cl::Device& device)
	{
		auto maxSizeOfStr = numOfBytes / 8;
		byte* decrypted = new byte[maxSizeOfStr]{};

		auto context = DeviceEx::CreateContext(device);
		auto bytesBuffer = CreateBuffer(context, bytes, numOfBytes, CL_MEM_READ_ONLY);
		auto decryptedBuffer = CreateBuffer(context, decrypted, maxSizeOfStr, CL_MEM_READ_WRITE);

		auto kernel = KernelEx::BuildFromFile(device, context, "Kernel.cl", "Decrypt");
		kernel.setArg(0, bytesBuffer);
		kernel.setArg(1, decryptedBuffer);

		auto queue = cl::CommandQueue(context, device);
		queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(maxSizeOfStr));
		queue.finish();

		ReadFromBuffer(queue, decrypted, maxSizeOfStr, decryptedBuffer);

		PrintDecrypted(decrypted);
		DestroyResources();
	}

	static void InitializeFromBin(const char* fileName)
	{
		std::ifstream fin(fileName, std::ios::binary);
		fin.read(reinterpret_cast<byte*>(&n), 4);
		fin.read(reinterpret_cast<byte*>(&m), 4);

		endswap(&n);
		endswap(&m);

		numOfBytes = 4 * m * n;
		bytes = new byte[numOfBytes];

		fin.read(bytes, numOfBytes);
		fin.close();
	}

	static void PrintEncryptedToBin(const char* fileName)
	{
		std::ofstream fout(fileName, std::ios::binary);

		endswap(&n);
		endswap(&m);

		fout.write(reinterpret_cast<byte*>(&n), sizeof(n));
		fout.write(reinterpret_cast<byte*>(&m), sizeof(m));

		fout.write(bytes, numOfBytes);
		fout.close();
	}

	static void PrintDecrypted(const char* decrypted) {
		std::cout << "Your message:" << std::endl;
		std::cout << decrypted << std::endl;
	}

	static void DestroyResources() {
		delete[] bytes;
	}

	static cl::Buffer CreateBuffer(const cl::Context& context, char* array, int size, int accessFlag)
	{
		return cl::Buffer(context, accessFlag | CL_MEM_COPY_HOST_PTR, size * sizeof(char), array);
	}

	static void ReadFromBuffer(const cl::CommandQueue& queue, char* array, int size, const cl::Buffer& buffer)
	{
		queue.enqueueReadBuffer(buffer, CL_TRUE, 0, size * sizeof(char), array);
	}
};

int Main::numOfBytes;
int Main::n;
int Main::m;
Main::byte* Main::bytes;


int main(int argc, char* argv[])
{
	Main::Run(argc, argv);
	system("pause");
}