#include <iostream>
#include <sys/stat.h>
using namespace std;
 
int main()
{
    // Path to the directory
    const char* dir = "../opencvextras/cars.mp4";
 
    // Structure which would store the metadata
    struct stat sb;
 
    // Calls the function with path as argument
    // If the file/directory exists at the path returns 0
    // If block executes if path exists
    if (stat(dir, &sb) == 0)
        cout << "The path is valid!";
    else
        cout << "The Path is invalid!";
 
    return 0;
}