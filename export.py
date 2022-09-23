
""" export MINDIR """
import argparse as arg
import numpy as np
import mindspore as ms
from mindspore import context, Tensor, export, load_checkpoint
from src.ghostnet import ghostnet_1x
from src.config import config


if __name__ == '__main__':
    parser = arg.ArgumentParser(description='SID export')
    parser.add_argument('--device_target', type=str, choices=['Ascend', 'GPU', 'CPU'], default='Ascend',
                        help='device where the code will be implemented')
    parser.add_argument('--device_id', type=int, default=0, help='device id')
    parser.add_argument('--file_format', type=str, choices=['AIR', 'MINDIR'], default='MINDIR',
                        help='file format')
    parser.add_argument('--checkpoint_path', required=True, default=None, help='ckpt file path')
    args = parser.parse_args()
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
    if args.device_target == 'Ascend':
        context.set_context(device_id=args.device_id)

    ckpt_dir = args.checkpoint_path
    net = ghostnet_1x(num_classes=config.num_classes)
    load_checkpoint(ckpt_dir, net=net)
    net.set_train(False)

    input_data = Tensor(np.zeros([1, 3, 224, 224]), ms.float32)
    print(input_data.shape)
    export(net, input_data, file_name='ghost', file_format=args.file_format)
