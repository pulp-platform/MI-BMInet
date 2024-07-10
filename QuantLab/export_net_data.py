import os
import numpy as np
import argparse
import json
import torch
import shutil

from torchsummary import summary
from main import main as quantlab_main

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--exp_id', help='experiment identification, e.g., 901', type=int, default=999)
parser.add_argument('-s', '--sample', help='index of the sample', type=int, default=0)
parser.add_argument('--train', help='Train network', action='store_true')
parser.add_argument('-a', '--all', help='Export all samples', action='store_true')
parser.add_argument('-p', '--print', help='Print model summary', action='store_true')
parser.add_argument('-f', '--expfolder', help='Experiment folder', default='.')
parser.add_argument('-d', '--dataset', help='Dataset {BCI-CompIV-2a, PhysionetMMMI}', default='PhysionetMMMI')
parser.add_argument('-n', '--network', help='Network {eegnet, edgeEEGNet}', default='edgeEEGNet')

args = parser.parse_args()

exp_folder = f'{args.dataset}/logs/{args.expfolder}/exp{args.exp_id:03}'
output_file = 'export/{}.npz'
output_config_file = "export/config.json"

# train the network
if args.train:
    # delete the exp folder
    try:
        shutil.rmtree(exp_folder)
        print('exp folder was deleted!')
    except:
        print('exp folder does not exist, skipping deletion')
    quantlab_main(f'{args.dataset}', f'{args.network}', exp_id=args.exp_id, ckpt_every=1, num_workers=1,
                  do_validPreTrain=False, use_single_gpu=True)

# import the edgeEEGnet folder
exec(open(f'quantlab/{args.dataset}/{args.network}/preprocess.py').read())
if 'edge' in args.network:
    exec(open(f'quantlab/{args.dataset}/{args.network}/edgeEEGnet.py').read())
else:
    exec(open(f'quantlab/{args.dataset}/{args.network}/eegnet.py').read())


exp_folder = f'{args.dataset}/logs/{args.expfolder}/exp{args.exp_id:03}'

# load the configuration file
with open(f'{exp_folder}/config.json') as _f:
    config = json.load(_f)

# get data loader
_, _, dataset = load_data_sets(f'{args.dataset}/data', config['treat']['data'])

nch=config['indiv']['net']['params']['C']
nsamples= config['indiv']['net']['params']['T']
print('number channels: ', nch, 'number samples: ', nsamples)

# load the model
ckpts = os.listdir(f'{exp_folder}/saves')
ckpts = [x for x in ckpts if "epoch" in x]
ckpts.sort()
last_epoch = int(ckpts[-1].replace('epoch', '').replace('.ckpt', ''))
ckpt = torch.load(f'{exp_folder}/saves/{ckpts[-1]}')
if 'edge' in args.network:
    model = edgeEEGNet(**config['indiv']['net']['params'])
else:
    model = EEGNet(**config['indiv']['net']['params'])
model.load_state_dict(ckpt['indiv']['net'])
for module in model.steController.modules:
    module.started = True

model.train(False)

# print summary
if args.print:
    model.cuda()
    summary(model, (1, nch, nsamples))
    # if "PhysionetMMMI" in args.dataset:
    #     summary(model, (1, nch, nsamples))
    # elif "BCI-CompIV-2a" in args.dataset:
    #     summary(model, (1, nch, nsamples))
    model.cpu()

# export all weights
weights = {key: value.cpu().detach().numpy() for key, value in ckpt['indiv']['net'].items()}
np.savez(output_file.format("net"), **weights)

if args.all:
    samples = []
    labels = []
    predictions = []

    n_samples = len(dataset)
    for sample in range(n_samples):
        x = dataset[sample][0]
        x = x.reshape(1, 1, nch, nsamples)
        label = dataset[sample][1]
        prediction = model(x)

        samples.append(x.numpy())
        labels.append(label.numpy())
        predictions.append(prediction.detach().numpy())

    np.savez(output_file.format("benchmark"), samples=samples, labels=labels, predictions=predictions)


# save input data
np.savez(output_file.format("input"), input=dataset[args.sample][0].numpy())

# prepare verification data
verification = {}
# do forward pass and compute the result of the network
with torch.no_grad():
    x = dataset[args.sample][0]
    verification['input'] = x.numpy()
    x = x.reshape(1, 1, nch, nsamples)
    # if "PhysionetMMMI" in args.dataset:
    #     x = x.reshape(1, 1, nch, nsamples)
    # elif "BCI-CompIV-2a" in args.dataset:
    #     x = x.reshape(1, 1, nch, nsamples)
    x = model.quant1(x)
    verification['input_quant'] = x.numpy()
    if not 'edge' in args.network:
        x = model.conv1_pad(x)
    x = model.conv1(x)
    verification['layer1_conv_out'] = x.numpy()
    x = model.batch_norm1(x)
    verification['layer1_bn_out'] = x.numpy()
    x = model.quant2(x)
    verification['layer1_activ'] = x.numpy()
    if 'edge' in args.network:
        x = model.conv2_pad(x)
    x = model.conv2(x)
    verification['layer2_conv_out'] = x.numpy()
    x = model.batch_norm2(x)
    verification['layer2_bn_out'] = x.numpy()
    x = model.activation1(x)
    verification['layer2_relu_out'] = x.numpy()
    x = model.pool1(x)
    verification['layer2_pool_out'] = x.numpy()
    x = model.quant3(x)
    verification['layer2_activ'] = x.numpy()
    x = model.sep_conv_pad(x)
    x = model.sep_conv1(x)
    verification['layer3_conv_out'] = x.numpy()
    x = model.quant4(x)
    verification['layer3_activ'] = x.numpy()
    x = model.sep_conv2(x)
    verification['layer4_conv_out'] = x.numpy()
    x = model.batch_norm3(x)
    verification['layer4_bn_out'] = x.numpy()
    x = model.activation2(x)
    verification['layer4_relu_out'] = x.numpy()
    x = model.pool2(x)
    verification['layer4_pool_out'] = x.numpy()
    x = model.quant5(x)
    verification['layer4_activ'] = x.numpy()
    x = model.flatten(x)
    x = model.fc(x)
    verification['output'] = x.numpy()
    x = model.quant6(x)
    verification['output_quant'] = x.numpy()

    print(f'sample {args.sample} - true label: {dataset[args.sample][1]}')

np.savez(output_file.format("verification"), **verification)

# copy the configuration file to the export folder
shutil.copyfile(f'{exp_folder}/config.json', output_config_file)
