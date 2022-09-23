# ghostnet

Deploying convolutional neural networks (CNNs) on embedded devices is difficult due to the limited memory and
computation resources.

[Paper](https://arxiv.org/pdf/1911.11907.pdf): Kai Han, Yunhe Wang, Qi Tian."GhostNet: More Features From Cheap Operations"

# Dataset

[ImageNet2012](http://www.image-net.org/)

```text
└─dataset
    ├─ilsvrc                  # Train
    └─validation_preprocess   # Eval
```

# Train

```bash
bash run_standalone_train.sh [DATASET_PATH] [PRETRAINED_CKPT_PATH])(Optional)
```

```text
epoch: 1 step: 1251, loss is 5.001419
epoch time: 457012.100 ms, per step time: 365.317 ms
epoch: 2 step: 1251, loss is 4.275552
epoch time: 280175.784 ms, per step time: 223.961 ms
epoch: 3 step: 1251, loss is 4.0788813
epoch time: 280134.943 ms, per step time: 223.929 ms
epoch: 4 step: 1251, loss is 4.0310946
epoch time: 280161.342 ms, per step time: 223.950 ms
epoch: 5 step: 1251, loss is 3.7326777
epoch time: 280178.602 ms, per step time: 223.964 ms
...
```

# Eval

```bash
bash run_eval.sh [DATASET_PATH] [CHECKPOINT_PATH]
```

```text
result: {'top_5_accuracy': 0.9162371134020618, 'top_1_accuracy': 0.739368556701031}
ckpt = /home/lzu/ghost_Mindspore/scripts/device0/ghostnet-500_1251.ckpt
```