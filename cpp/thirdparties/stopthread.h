
#include <chrono>
#include <mutex>
#include <future>
#include <thread>

int getDateTime(char * psDate){
    time_t nSeconds;
    struct tm *pTM;
    
    time(&nSeconds);
    pTM = localtime(&nSeconds);
    // "%Y-%m-%d %H:%M:%S"
    sprintf(psDate,"%04d-%02d-%02d %02d:%02d:%02d", 
            pTM->tm_year + 1900, pTM->tm_mon + 1, pTM->tm_mday,
            pTM->tm_hour, pTM->tm_min, pTM->tm_sec);
    return 0;
}

class StopThread {
public:
    StopThread() {}
    ~StopThread() {}

    void run() {
        mThread = std::thread([this]() {
            this->threadLoop(this->mExitSignal.get_future());
        });
        mThread.detach();
    }

    void join() {
        mThread = std::thread([this]() {
            this->threadLoop(this->mExitSignal.get_future());
        });
        mThread.join();
    }

    void stop() {
        mExitSignal.set_value();
    }

private:
    std::thread mThread;
    std::promise<void> mExitSignal;

    virtual void threadLoop(std::future<void> exitListener) {
        do {
            // do something here
        } while (exitListener.wait_for(std::chrono::microseconds(1)) == std::future_status::timeout);
    }
};
