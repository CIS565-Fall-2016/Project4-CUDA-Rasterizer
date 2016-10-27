
#include <cstdio>
#include <chrono>

#include "timer.h"

#include <cuda.h>
#include <cuda_runtime.h>

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
static void checkCUDAErrorFn(const char *msg, const char *file, int line) {
#if ERRORCHECK
	cudaDeviceSynchronize();
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess == err) {
		return;
	}

	fprintf(stderr, "CUDA error");
	if (file) {
		fprintf(stderr, " (%s:%d)", file, line);
	}
	fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#  ifdef _WIN32
	getchar();
#  endif
	exit(EXIT_FAILURE);
#endif
}

MyTimer* Timer::m_myTimer = NULL;

class MyTimer
{
public:
	using Clock = std::chrono::high_resolution_clock;
	using TimePoint = std::chrono::time_point<Clock>;

public:
	MyTimer()
	{
		m_refCount = 0;
		m_useGPU = true;
		m_elapsedTimeInms = 0.0f;

		cudaEventCreate(&m_start);
		cudaEventCreate(&m_stop);

		m_startTime = Clock::now();
		m_stopTime = Clock::now();
	}

	~MyTimer()
	{
		cudaEventDestroy(m_start);
		cudaEventDestroy(m_stop);
	}

public:

	void resetTimer(bool useGPU = true) 
	{
		m_useGPU = useGPU;
		m_elapsedTimeInms = 0.0f;
	}

	void playTimer()
	{
		if (m_refCount++ == 0)
		{
			if (m_useGPU)
			{
				cudaEventRecord(m_start);
			}
			else
			{
				m_startTime = Clock::now();
			}
		}
	}

	bool pauseTimer()
	{
		bool bPaused = false;
		if (--m_refCount == 0)
		{
			float newElapsedTime = 0.0f;
			if (m_useGPU)
			{
				cudaEventRecord(m_stop);
				cudaEventSynchronize(m_stop);
				cudaEventElapsedTime(&newElapsedTime, m_start, m_stop);
			}
			else
			{
				m_stopTime = Clock::now();
				newElapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(m_stopTime - m_startTime).count();
			}
			m_elapsedTimeInms += newElapsedTime;
			bPaused = true;
		}
		return bPaused;
	}

	float printTimer(const char* timerHeader, float timerFactor)
	{
		float elapsedTime = timerFactor * m_elapsedTimeInms;
		printf("%s - Elapsed Time:%f ms.\n", timerHeader, elapsedTime);
		return elapsedTime;
	}

private:
	size_t m_refCount;
	bool m_useGPU;
	float m_elapsedTimeInms;
private:
	cudaEvent_t m_start;
	cudaEvent_t m_stop;

private:
	TimePoint m_startTime;
	TimePoint m_stopTime;
};

Timer::Timer()
{
}

Timer::~Timer()
{
}

void Timer::initializeTimer()
{
	if (m_myTimer == NULL)
		m_myTimer = new MyTimer;
}

void Timer::shutdownTimer()
{
	if (m_myTimer != NULL)
		delete m_myTimer;
}

void Timer::resetTimer(bool useGPU)
{
	m_myTimer->resetTimer(useGPU);
}

void Timer::playTimer()
{
	m_myTimer->playTimer();
}

void Timer::pauseTimer()
{
	m_myTimer->pauseTimer();
}

void Timer::printTimer(const char* timerHeader, float timerFactor)
{
	m_myTimer->printTimer(timerHeader, timerFactor);
}