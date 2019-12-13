__kernel void SM(__global double* firstM,
				 __global double* secondM,
				 __global double* answerM,
				 int n, int m, int l)
{
	int i = get_global_id(0);
	int j = get_global_id(1);

	double resultSumma = 0;
	int firstMBase = i * m;
	int secondMBase = j * m;

	for (int k = 0; k < m; ++k)
	{
		resultSumma += firstM[firstMBase + k] * secondM[secondMBase + k];
	}

	answerM[i * l + j] = resultSumma;
}

__kernel void V4M(__global double* firstM,
				  __global double* secondM,
				  __global double* answerM,
				  int n, int m, int l)
{
	int i = get_global_id(0);
	int j = get_global_id(1);

	double4 resultSumma4 = (double4)(0);
	int firstMBase = i * m;
	int secondMBase = j * m;
	int reps = m / 4;

	for (int k = 0; k < reps; ++k)
	{
		double4 v1 = (double4)(firstM[firstMBase + 4 * k], firstM[firstMBase + 1 + 4 * k], firstM[firstMBase + 2 + 4 * k], firstM[firstMBase + 3 + 4 * k]);
		double4 v2 = (double4)(secondM[secondMBase + 4 * k], secondM[secondMBase + 1 + 4 * k], secondM[secondMBase + 2 + 4 * k], secondM[secondMBase + 3 + 4 * k]);
		resultSumma4 += v1 * v2; // dot(v1, v2) ?
	}

	int b4 = reps * 4;

	firstMBase += b4;
	secondMBase += b4;
	reps = m - b4;
	double resultSumma = 0;

	for (int k = 0; k < reps; ++k)
	{
		resultSumma += firstM[firstMBase + k] * secondM[secondMBase + k];
	}

	answerM[i * l + j] = resultSumma4.s0 + resultSumma4.s1 + resultSumma4.s2 + resultSumma4.s3 + resultSumma;
}


__kernel void V8M(__global double* firstM,
				  __global double* secondM,
				  __global double* answerM,
				  int n, int m, int l)
{
	int i = get_global_id(0);
	int j = get_global_id(1);

	double8 resultSumma = (double8)(0);
	int firstMBase = i * m;
	int secondMBase = j * m;
	int reps = m / 8;

	for (int k = 0; k < reps; ++k)
	{
		double8 v1 = (double8)(firstM[firstMBase + 8 * k], firstM[firstMBase + 1 + 8 * k], firstM[firstMBase + 2 + 8 * k], firstM[firstMBase + 3 + 8 * k], firstM[firstMBase + 4 + 8 * k], firstM[firstMBase + 5 + 8 * k], firstM[firstMBase + 6 + 8 * k], firstM[firstMBase + 7 + 8 * k]);
		double8 v2 = (double8)(secondM[secondMBase + 8 * k], secondM[secondMBase + 1 + 8 * k], secondM[secondMBase + 2 + 8 * k], secondM[secondMBase + 3 + 8 * k], secondM[secondMBase + 4 + 8 * k], secondM[secondMBase + 5 + 8 * k], secondM[secondMBase + 6 + 8 * k], secondM[secondMBase + 7 + 8 * k]);
		resultSumma += v1 * v2;
	}

	answerM[i * l + j] = resultSumma.s0 + resultSumma.s1 + resultSumma.s2 + resultSumma.s3 + resultSumma.s4 + resultSumma.s5 + resultSumma.s6 + resultSumma.s7;
}

__kernel void V16M(__global double* firstM,
				   __global double* secondM,
				   __global double* answerM,
				   int n, int m, int l)
{
	int i = get_global_id(0);
	int j = get_global_id(1);

	double16 resultSumma16 = (double16)(0);
	int firstMBase = i * m;
	int secondMBase = j * m;
	int reps16 = m / 16;

	for (int k = 0; k < reps16; ++k)
	{
		double16 v1 = (double16)(firstM[firstMBase + 16 * k], firstM[firstMBase + 1 + 16 * k], firstM[firstMBase + 2 + 16 * k], firstM[firstMBase + 3 + 16 * k], firstM[firstMBase + 4 + 16 * k], firstM[firstMBase + 5 + 16 * k], firstM[firstMBase + 6 + 16 * k], firstM[firstMBase + 7 + 16 * k], firstM[firstMBase + 8 + 16 * k], firstM[firstMBase + 9 + 16 * k], firstM[firstMBase + 10 + 16 * k], firstM[firstMBase + 11 + 16 * k], firstM[firstMBase + 12 + 16 * k], firstM[firstMBase + 13 + 16 * k], firstM[firstMBase + 14 + 16 * k], firstM[firstMBase + 15 + 16 * k]);
		double16 v2 = (double16)(secondM[secondMBase + 16 * k], secondM[secondMBase + 1 + 16 * k], secondM[secondMBase + 2 + 16 * k], secondM[secondMBase + 3 + 16 * k], secondM[secondMBase + 4 + 16 * k], secondM[secondMBase + 5 + 16 * k], secondM[secondMBase + 6 + 16 * k], secondM[secondMBase + 7 + 16 * k], secondM[secondMBase + 8 + 16 * k], secondM[secondMBase + 9 + 16 * k], secondM[secondMBase + 10 + 16 * k], secondM[secondMBase + 11 + 16 * k], secondM[secondMBase + 12 + 16 * k], secondM[secondMBase + 13 + 16 * k], secondM[secondMBase + 14 + 16 * k], secondM[secondMBase + 15 + 16 * k]);
		resultSumma16 += v1 * v2;
	}

	int b16 = reps16 * 16;

	double resultSumma = 0;
	firstMBase += b16;
	secondMBase += b16;
	int reps = m - b16;

	for (int k = 0; k < reps; ++k)
	{
		resultSumma += firstM[firstMBase + k] * secondM[secondMBase + k];
	}

	answerM[i * l + j] = resultSumma16.s0 + resultSumma16.s1 + resultSumma16.s2 + resultSumma16.s3 + resultSumma16.s4 + resultSumma16.s5 + resultSumma16.s6 + 
		resultSumma16.s7 + resultSumma16.s8 + resultSumma16.s9 + resultSumma16.sa + resultSumma16.sb + resultSumma16.sc + resultSumma16.sd + resultSumma16.se + 
		resultSumma16.sf + resultSumma;
	//answerM[i * l + j] = dot(resultSumma16, double16(1)); ?
}