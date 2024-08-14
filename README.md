# PA
This is a markdown file test.
## Environment

* Python (3.7.10)
* Pytorch (1.7.1)
* torchvision (0.8.2)
* CUDA
* Numpy

## File Structure
```
datasets
├──cifar100
PADE-master
├── README.md
├── ...                                
```
## Training
```
python train_cifar.py --lr 0.1 --epochs 200 --lam 0.5 --t 1. --pa True --resume None
```
## Evaluating
```
python test_all_cifar.py --resume log/cifar100_resnet32_3_0.01_4/checkpoint.best.pth.tar --num_experts 3 --num_classes 100 --pa True
```
