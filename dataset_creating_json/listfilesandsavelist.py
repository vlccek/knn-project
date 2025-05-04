
import os

def main():
    files = os.listdir("./dataset_obsahy/images/")
    for i in files:

        print(i)


if __name__ == '__main__':
    main()