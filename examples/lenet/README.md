# LeNet

Data for  LeNet can be downloaded as follows:

```bash
./download_mnist.sh
```

Weights produced by training the model in Torch have been provided.
The training script (closely following the one found [here](https://github.com/Elman295/Paper_with_code/blob/eae081e2be38680e034a3e7ca3075b2360911953/LeNet_5_Pytorch.ipynb)) is [provided](lenet.py).

To run LeNet inference:

```bash
cd ../../build
./examples/lenet/lenet ../examples/lenet/weights ../examples/lenet/data
```
