import os
 
if __name__ == '__main__':
 
    for x in os.environ:
        print((x, os.getenv(x)))
        