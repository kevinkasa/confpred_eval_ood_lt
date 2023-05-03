"""
Run inference on different neural nets / datasets, and save softmax scores.
"""
import argparse

import torch
import torchvision
import timm

import get_data


def get_args_parser():
    parser = argparse.ArgumentParser('Script for saving inference results from PyTorch models', add_help=False)

    parser.add_argument('--exp_name', type=str, help='Experiment name')
    parser.add_argument('--model', type=str, choices=['deit3B', 'deit3S', 'vitB', 'vitS', 'resnet152', 'resnet50'],
                        help='Model name')
    parser.add_argument('--datasets', type=str, nargs='+', help='List of datasets (IN1k, INv2, etc)')
    return parser


def main(args):
    supported_datasets = [
        'INk1',
        'INv2',
        'INa',
        'INc',
        'INr',
        'INw',
    ]

    timm_models = ['deit3B', 'deit3S', 'vitB', 'vitS', ]
    torch_models = ['resnet152', 'resnet50', ]

    # create the model
    if args.model == 'deit3B':
        model = timm.create_model(
            'deit3_base_patch16_224', pretrained=True)
        print('Loaded Deit3-B')
    elif args.model == 'deit3S':
        model = timm.create_model(
            'deit3_small_patch16_224', pretrained=True)
        print('Loaded Deit3-S')
    elif args.model == 'vitB':
        model = timm.create_model(
            'vit_base_patch16_224.augreg_in21k_ft_in1k', pretrained=True)
        print('Loaded ViT-B')
    elif args.model == 'vitS':
        model = timm.create_model(
            'vit_small_patch16_224.augreg_in21k_ft_in1k', pretrained=True)
        print('Loaded ViT-S')
    elif args.model == 'resnet152':
        model = torchvision.models.resnet152(pretrained=True, progress=True)
        print('Loaded ResNet-152')
    elif args.model == 'resnet50':
        model = torchvision.models.resnet50(pretrained=True, progress=True)
        print('Loaded ResNet-50')
    else:  # TODO: support torchvision resnet-50/152
        raise ValueError('Invalid model selected')

    model = model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # check if model is from timm or torchvision for loading transforms
    if args.model in timm_models:  # Model is from timm
        data_config = timm.data.resolve_model_data_config(model)
        transform = timm.data.create_transform(**data_config, is_training=False)

    elif args.model in torch_models:  # Model is from torchvision
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        raise ValueError("Unknown model - loading transformations failed")

    for data in args.datasets:
        if data not in supported_datasets:
            raise ValueError('Dataset not supported')

        getattr(get_data, data)(exp_name=args.exp_name, transform=transform, model=model)

    print('Finished saving results')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Script for saving inference results from PyTorch models',
                                     parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
