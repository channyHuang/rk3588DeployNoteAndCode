#include <iostream>

#include "threadPool.h"

ThreadPool pool(3);

struct stFrame {
    int nWidth;
    int nHeight;
};

int main(int argc, char** argv) {
    std::queue< std::future<stFrame> > results;
    int index = 0;
    while (true) {
        results.push(
            pool.enqueue([&index] {
                stFrame frame;
                frame.nHeight = ++index;
                frame.nWidth = ++index;
                std::cout << "pool enqueue " << frame.nHeight << " " << frame.nWidth << std::endl;
                return frame;
            })
        );

        while (!results.empty()) {
            auto &result = results.front();

            stFrame frame = result.get();
            std::cout << frame.nHeight << " " << frame.nWidth << std::endl;
            results.pop();
        }
    }
        
    return 0;
}
