#pragma once

class MyTimer;

class Timer
{
public:
	Timer();
	~Timer();

public:
	static void initializeTimer();
	static void shutdownTimer();

public:
	static void resetTimer(bool useGPU = true);
	static void playTimer();
	static void pauseTimer();
	static void printTimer(const char* timerHeader, float timerFactor);

private:
	static MyTimer* m_myTimer;
};
