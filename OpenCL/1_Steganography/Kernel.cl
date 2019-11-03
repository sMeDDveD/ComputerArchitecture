__kernel void Encrypt(__global char* bytes,
					  __global char* toEncrypt)
{
	int index = get_global_id(0);
	char cSymbol = toEncrypt[index];

	for (int i = 0; i < 8; ++i)
	{
		bytes[index * 8 + i] = (bytes[index * 8 + i] | 1) & ~(1 ^ (cSymbol & (1 << i) ? 1 : 0));
	}
}

__kernel void Decrypt(__global char* bytes,
					  __global char* toDecrypt)
{
	int index = get_global_id(0);

	for (int i = 0; i < 8; i++)
	{ 
		toDecrypt[index] |= ((bytes[index * 8 + i] & 1) ? 1 : 0) << i;
	}
}