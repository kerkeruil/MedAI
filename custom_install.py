import subprocess
import sys

def install(name):
    subprocess.call([sys.executable, '-m', 'pip', 'install', name])


if __name__ == '__main__':
    install('accelerate')
    install('transformers')
    install('datasets')
    install('torchvision')
    install('tensorboardX')
    install('nibabel')
