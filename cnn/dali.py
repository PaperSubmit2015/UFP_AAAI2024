import argparse
import os
import random
import shutil
import time
import warnings
import math

#from datetime import  datetime 

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from   torch.utils.dlpack import  to_dlpack, from_dlpack
from   torch.utils.cpp_extension import load

import  cupy as cp
import  numpy as np
import  plot
import  pickle
import  lmdb

from  PIL import  Image


try:
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy
    from nvidia.dali.pipeline import pipeline_def
    import nvidia.dali.types as types
    import nvidia.dali.fn as fn
    from nvidia.dali.backend import TensorListGPU, TensorGPU, TensorCPU, TensorListCPU
    #from nvidia.dali.dependency import TensorNode
except ImportError:
    raise ImportError("Please install DALI from https://www.github.com/NVIDIA/DALI to run this example.")


#from folder2lmdb import ImageFolderLMDB

import qmethod  as qm
import qTensor  as qt
from   quant_model import *

from  record import  showCurve,  recordCurve

import  utils.envirionment  as env
from  utils.SaveImage import  TensorToFile
from  dbFormat import  *

build_directory = os.getcwd()
build_directory = os.getcwd().replace("\\","/" )
libqt     = load( 'quant', [fqt ], verbose=True,
                   build_directory = build_directory)

args =None  
best_acc1 = 0




def parse():
    model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('data', metavar='DIR', nargs='*',
                        help='path(s) to dataset (if one path is provided, it is assumed\n' +
                       'to have subdirectories named "train" and "val"; alternatively,\n' +
                       'train and val paths can be specified directly by providing both paths as arguments)')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                        choices=model_names,
                        help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N', help='mini-batch size per process (default: 256)')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='Initial learning rate.  Will be scaled by <global batch size>/256: args.lr = args.lr*float(args.batch_size*args.world_size)/256.  A warmup schedule will also be applied over the first 5 epochs.')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')

    parser.add_argument('--dali_cpu', action='store_true',
                        help='Runs CPU based version of DALI pipeline.')
    parser.add_argument('--prof', default=-1, type=int,
                        help='Only run 10 iterations for profiling.')
    parser.add_argument('--deterministic', action='store_true')

    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument('--sync_bn', action='store_true',
                        help='enabling apex sync BN.')

    parser.add_argument('--opt-level', type=str, default=None)
    parser.add_argument('--keep-batchnorm-fp32', type=str, default=None)
    parser.add_argument('--loss-scale', type=str, default=None)
    parser.add_argument('--channels-last', type=bool, default=False)
    parser.add_argument('-t', '--test', action='store_true',
                        help='Launch test mode with preset arguments')
    parser.add_argument('--qwm', default="none", type=str,
                    choices= quantSchemes  ,
                     help='Apply quantization methods on weight or activation. '
                          'The quantization methods includes uq, eci, cdf,lloyd  ')

    parser.add_argument('--qwd', default=8, type=int,
                     help='Denote the quantization level in bits. If it is 0, it will search the optimal bit depth.'
                          '0 : using floating point.')

    parser.add_argument('--qwe', default=0, type=int,
                     help='Denote the E field of quantization number(EpMq,ECI). If it is 0, it will search the optimal bit depth.'
                          '0 : exploring the best E value .'
                          )
    parser.add_argument('--qwsr', default="self", type=str,
                     help='Define  stocastic rounding mode  ')
                          
    parser.add_argument('--dataloader', type=str, default="file")

    parser.add_argument('--gpu', default=0, type=int, metavar='N',
                        help='gpu device to run')

    parser.add_argument('--qgm', default="none", type=str,
                    choices= quantSchemes ,
                     help='Apply quantization methods on gradient. '
                          'The quantization methods includes uq, eci, cdf,lloyd. Default is none(floating point). ')
    parser.add_argument('--qgd', default=8, type=int,
                     help='Denote gradient quantization level. If it is negtive, it will search the optimal bit depth.'
                          '0 : using floating point.')
    
    parser.add_argument('--qge', default=2, type=int,
                     help='Denote the gradient aquantization level(bits). If it is negtive, it will search the optimal bit depth.'
                          '0 : using floating point.')

    parser.add_argument('--qgsr', default="self", type=str,
                     help='Define  stocastic rounding mode  ')

    parser.add_argument('--qam', default="none", type=str,
                    choices= quantSchemes ,
                     help='Apply quantization methods on weight or activation. '
                          'The quantization methods includes uq, eci, cdf,lloyd. Default is none(floating point). ')

    parser.add_argument('--qad', default=8, type=int,
                     help='Denote the quantization level in bits. If it is 0, it will search the optimal bit depth.'
                          '0 : using floating point.')
                          
    parser.add_argument('--qae', default=2, type=int,
                     help='Denote the quantization level in bits. If it is 0, it will search the optimal bit depth.'
                          '0 : using floating point.')

    parser.add_argument('--qasr', default="self", type=str,
                     help='Define  stocastic rounding mode  ')

                          
    parser.add_argument('--probeAE', default=0, type=int,
                     help='Denote the quantization level in bits. If it is 0, it will search the optimal bit depth.')
                          
    parser.add_argument('--profile', default=0, type=int,
                     help='Denote the quantization level in bits. If it is 0, it will search the optimal bit depth.')

    parser.add_argument('--qL0', default=0, type=int,
                     help='Denote whether to quant the first layer(L0) . If it is 1, the first layer will be quantized.')

    parser.add_argument('--qLL', default=0, type=int,
                     help='Denote whether to quant the last layer(LL). If it is 1, the last layer(LL) will be quantized. ')
                          
    parser.add_argument('--needProbeE', default=0, type=int,
                     help='Denote whether to quant the last FC layer. If it is 0, it will search the optimal bit depth.')
    parser.add_argument('--qat', default='none', type=str,
                     help='Apply quantization aware training.  '
                          'The quantization methods includes uq, eci, cdf,lloyd  ')

    parser.add_argument('--logpath', default='./logs', type=str,
                     help='Direcotry  for storing log infomation.  ')
#    parser.add_argument('--train_with_eci', default=False, type=  bool , 
 #                    help='Train with ECI8 ' )

    parser.add_argument('--verbose', default=0, type=int,    help='verbose. ')   

    parser.add_argument('--qcfg', default="", type=str,    help='quantization config file. ')   
    parser.add_argument('-q', '--qscheme', metavar='QUANT', default='none', 
                    choices= ['none','float','ptq','qat'] ,
                    help=' Quantization scheme: none/float, ptq(post-training), qat(quantization aware) ')

    parser.add_argument('--profileDataPercent', default=1, type=int,
                     help='Denote the fraction that is needed for activation profiling in wtlloyd quantization. It must be within [1,100]. Default value is  10.' )

    parser.add_argument('--statInterval', default=30, type=int,
                     help='Denote whether to quant the last layer(LL). If it is 1, the last layer(LL) will be quantized. ')

    args = parser.parse_args()

    if  args.qam != "none"  or  args.qwm != "none":
        args.FQT = True
    elif args.qgm != 'none' :
        args.FQT = True
    else:
        args.FQT = False

    return args


# item() is a recent addition, so this helps with backward compatibility.
def to_python_float(t):
    if hasattr(t, 'item'):
        return t.item()
    else:
        return t[0]




@pipeline_def
def create_lmdb_pipeline(dataset,  crop, size, shard_id, num_shards, dali_cpu=False, is_training=True):

    
    jpegs , labels = fn.external_source(source= dataset,
                                        num_outputs=2)
                                        #, dtype=types.UINT8)
    images = jpegs                                        

    dali_device = 'cpu' if dali_cpu else 'gpu'
    decoder_device = 'cpu' if dali_cpu else 'mixed'
    # ask nvJPEG to preallocate memory for the biggest sample in ImageNet for CPU and GPU to avoid reallocations in runtime
    device_memory_padding = 211025920 if decoder_device == 'mixed' else 0
    host_memory_padding = 140544512 if decoder_device == 'mixed' else 0
    # ask HW NVJPEG to allocate memory ahead for the biggest image in the data set to avoid reallocations in runtime
    preallocate_width_hint = 5980 if decoder_device == 'mixed' else 0
    preallocate_height_hint = 6430 if decoder_device == 'mixed' else 0

    if is_training:
        images = fn.decoders.image_random_crop(images,
                                               device=decoder_device, output_type=types.RGB,
                                               device_memory_padding=device_memory_padding,
                                               host_memory_padding=host_memory_padding,
                                               preallocate_width_hint=preallocate_width_hint,
                                               preallocate_height_hint=preallocate_height_hint,
                                               random_aspect_ratio=[0.8, 1.25],
                                               random_area=[0.1, 1.0],
                                               num_attempts=100)
        images = fn.resize(images,
                           device=dali_device,
                           resize_x=crop,
                           resize_y=crop,
                           interp_type=types.INTERP_TRIANGULAR)
        mirror = fn.random.coin_flip(probability=0.5)
    else:
        images = fn.decoders.image(images,
                                   device=decoder_device,
                                   output_type=types.RGB)
        images = fn.resize(images,
                           device=dali_device,
                           size=size,
                           mode="not_smaller",
                           interp_type=types.INTERP_TRIANGULAR)
        mirror = False

    images = fn.crop_mirror_normalize(images.gpu(),
                                      dtype=types.FLOAT,
                                      output_layout="CHW",
                                      crop=(crop, crop),
                                      mean=[0.485 * 255,0.456 * 255,0.406 * 255],
                                      std=[0.229 * 255,0.224 * 255,0.225 * 255],
                                      mirror=mirror)

    labels = labels.gpu()
    return images, labels

@pipeline_def
def create_dali_pipeline(data_dir, crop, size, shard_id, num_shards, dali_cpu=False, is_training=True):
    images, labels = fn.readers.file(file_root=data_dir,
                                     shard_id=shard_id,
                                     num_shards=num_shards,
                                     random_shuffle=is_training,
                                     pad_last_batch=True,
                                     name="Reader")

    dali_device = 'cpu' if dali_cpu else 'gpu'
    decoder_device = 'cpu' if dali_cpu else 'mixed'

    # ask nvJPEG to preallocate memory for the biggest sample in ImageNet for CPU and GPU to avoid reallocations in runtime
    device_memory_padding = 211025920 if decoder_device == 'mixed' else 0
    host_memory_padding = 140544512 if decoder_device == 'mixed' else 0
    # ask HW NVJPEG to allocate memory ahead for the biggest image in the data set to avoid reallocations in runtime
    preallocate_width_hint = 5980 if decoder_device == 'mixed' else 0
    preallocate_height_hint = 6430 if decoder_device == 'mixed' else 0
    if is_training:
        images = fn.decoders.image_random_crop(images,
                                               device=decoder_device, output_type=types.RGB,
                                               device_memory_padding=device_memory_padding,
                                               host_memory_padding=host_memory_padding,
                                               preallocate_width_hint=preallocate_width_hint,
                                               preallocate_height_hint=preallocate_height_hint,
                                               random_aspect_ratio=[0.8, 1.25],
                                               random_area=[0.1, 1.0],
                                               num_attempts=100)
        images = fn.resize(images,
                           device=dali_device,
                           resize_x=crop,
                           resize_y=crop,
                           interp_type=types.INTERP_TRIANGULAR)
        mirror = fn.random.coin_flip(probability=0.5)
    else:
        images = fn.decoders.image(images,
                                   device=decoder_device,
                                   output_type=types.RGB)
        images = fn.resize(images,
                           device=dali_device,
                           size=size,
                           mode="not_smaller",
                           interp_type=types.INTERP_TRIANGULAR)
        mirror = False
    
    images = fn.crop_mirror_normalize(images.gpu(),
                                      dtype=types.FLOAT,
                                      output_layout="CHW",
                                      crop=(crop, crop),
                                      mean=[0.485 * 255,0.456 * 255,0.406 * 255],
                                      std=[0.229 * 255,0.224 * 255,0.225 * 255],
                                      mirror=mirror)

    labels = labels.gpu()

    return images, labels



@pipeline_def
def create_bing_pipeline(dataset, crop, size, shard_id, num_shards, dali_cpu=False, is_training=True):
    """
    images, labels = fn.readers.file(file_root=data_dir,
                                     shard_id=shard_id,
                                     num_shards=num_shards,
                                     random_shuffle=is_training,
                                     pad_last_batch=True,
                                     name="Reader")
    """    
    images , labels = fn.external_source(source=dataset, num_output=2)

    dali_device = 'cpu' if dali_cpu else 'gpu'
    decoder_device = 'cpu' if dali_cpu else 'mixed'

    # ask nvJPEG to preallocate memory for the biggest sample in ImageNet for CPU and GPU to avoid reallocations in runtime
    device_memory_padding = 211025920 if decoder_device == 'mixed' else 0
    host_memory_padding = 140544512 if decoder_device == 'mixed' else 0
    # ask HW NVJPEG to allocate memory ahead for the biggest image in the data set to avoid reallocations in runtime
    preallocate_width_hint = 5980 if decoder_device == 'mixed' else 0
    preallocate_height_hint = 6430 if decoder_device == 'mixed' else 0
    if is_training:
        images = fn.decoders.image_random_crop(images,
                                               device=decoder_device, output_type=types.RGB,
                                               device_memory_padding=device_memory_padding,
                                               host_memory_padding=host_memory_padding,
                                               preallocate_width_hint=preallocate_width_hint,
                                               preallocate_height_hint=preallocate_height_hint,
                                               random_aspect_ratio=[0.8, 1.25],
                                               random_area=[0.1, 1.0],
                                               num_attempts=100)
        images = fn.resize(images,
                           device=dali_device,
                           resize_x=crop,
                           resize_y=crop,
                           interp_type=types.INTERP_TRIANGULAR)
        mirror = fn.random.coin_flip(probability=0.5)
    else:
        images = fn.decoders.image(images,
                                   device=decoder_device,
                                   output_type=types.RGB)
        images = fn.resize(images,
                           device=dali_device,
                           size=size,
                           mode="not_smaller",
                           interp_type=types.INTERP_TRIANGULAR)
        mirror = False
    
    images = fn.crop_mirror_normalize(images.gpu(),
                                      dtype=types.FLOAT,
                                      output_layout="CHW",
                                      crop=(crop, crop),
                                      mean=[0.485 * 255,0.456 * 255,0.406 * 255],
                                      std=[0.229 * 255,0.224 * 255,0.225 * 255],
                                      mirror=mirror)

    labels = labels.gpu()

    return images, labels




class  LMDB_iter():
    def  __init__(self, pipe, total):
        self.pipe=pipe
        self._size = total
        self.pipe.reset()

    def reset(self):
        self.pipe.reset()

    def  __next__(self):

        try:
            out= self.pipe.run()
        except StopIteration:
            raise StopIteration

        images = out[0]
        target = out[1] 

        images = images.as_tensor()
        images = cp.asanyarray (images)
        images = from_dlpack(images)

        target = target.as_tensor()
        target = cp.asanyarray (target)
        target = from_dlpack(target)
    
        out ={'data' : images,  'label' :target  } 
        return  [out]

    def __iter__(self):
        return  self



def main():
    global best_prec1, args
    best_prec1 = 0
    args = parse()

    print(args)

    # record starttime and jobname
    start_time = time.localtime()  
    qm = "-W" + args.qwm +str(args.qwd) +'-A' + args.qam + str(args.qad)  +"-"
    ftime = time.strftime("%Y%m%d_%H-%M", start_time )
    jobname = args.arch  + qm +  ftime   


    # test mode, use default args for sanity test
    if args.test:
        args.opt_level = None
        args.epochs =   90
        #args.start_epoch = 0
        args.arch = 'resnet18'
        args.batch_size = 64
        #args.data = []
        args.sync_bn = False
        #args.data.append('/data/imagenet/train-jpeg/')
        #args.data.append('/data/imagenet/val-jpeg/')
        print("Test mode - no DDP, no apex, RN50, 10 iterations")
        print(" args.dataloader =", args.dataloader)

    if not len(args.data):
        raise Exception("error: No data set provided")

    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1

    # make apex optional
    if args.opt_level is not None or args.distributed or args.sync_bn:
        try:
            global DDP, amp, optimizers, parallel
            from apex.parallel import DistributedDataParallel as DDP
            from apex import amp, optimizers, parallel
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")

    print("opt_level = {}".format(args.opt_level))
    print("keep_batchnorm_fp32 = {}".format(args.keep_batchnorm_fp32), type(args.keep_batchnorm_fp32))
    print("loss_scale = {}".format(args.loss_scale), type(args.loss_scale))

    print("\nCUDNN VERSION: {}\n".format(torch.backends.cudnn.version()))

    cudnn.benchmark = True
    best_prec1 = 0
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.manual_seed(args.local_rank)
        torch.set_printoptions(precision=10)


    args.world_size = 1
    torch.cuda.set_device(args.gpu)

    if args.distributed:
        args.gpu = args.local_rank
        torch.cuda.set_device(args.gpu)
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://')
        args.world_size = torch.distributed.get_world_size()

    args.total_batch_size = args.world_size * args.batch_size
    assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."

    torch.cuda.set_device(args.gpu)
    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()


    if args.sync_bn:
        print("using apex synced BN")
        model = parallel.convert_syncbn_model(model)

    if hasattr(torch, 'channels_last') and  hasattr(torch, 'contiguous_format'):
        if args.channels_last:
            memory_format = torch.channels_last
        else:
            memory_format = torch.contiguous_format
        model = model.cuda().to(memory_format=memory_format)
    else:
        model = model.cuda()

    # Scale learning rate based on global batch size
    args.lr = args.lr*float(args.batch_size*args.world_size)/256.
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # Initialize Amp.  Amp accepts either values or strings for the optional override arguments,
    # for convenient interoperation with argparse.
    if args.opt_level is not None:
        model, optimizer = amp.initialize(model, optimizer,
                                          opt_level=args.opt_level,
                                          keep_batchnorm_fp32=args.keep_batchnorm_fp32,
                                          loss_scale=args.loss_scale
                                          )

    # For distributed training, wrap the model with apex.parallel.DistributedDataParallel.
    # This must be done AFTER the call to amp.initialize.  If model = DDP(model) is called
    # before model, ... = amp.initialize(model, ...), the call to amp.initialize may alter
    # the types of model's parameters in a way that disrupts or destroys DDP's allreduce hooks.
    if args.distributed:
        # By default, apex.parallel.DistributedDataParallel overlaps communication with
        # computation in the backward pass.
        # model = DDP(model)
        # delay_allreduce delays all communication to the end of the backward pass.
        model = DDP(model, delay_allreduce=True)



    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    curve = [] # record training curve
    # Optionally resume from a checkpoint
    if args.resume:
        # Use a local scope to avoid dangling references
        def resume():
            if os.path.isfile(args.resume):
                print("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume, map_location = lambda storage, loc: storage.cuda(args.gpu))
                args.start_epoch = checkpoint['epoch']
                global best_prec1
                #best_prec1 = checkpoint['best_prec1']
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(args.resume, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(args.resume))
        resume()

    # Data loading code

    if  args.dataloader == "lmdb" :
        traindir = os.path.join(args.data[0], 'lmdb')
        valdir = os.path.join(args.data[0], 'lmdb')
    elif  args.dataloader == "bing" :
        traindir = os.path.join(args.data[0], 'bing')
        valdir = os.path.join(args.data[0], 'bing')


    elif  args.dataloader == "file" :
        traindir = os.path.join(args.data[0], 'imagenet')
        valdir = os.path.join(args.data[0], 'imagenet')

    if len(args.data) == 1:
        traindir = os.path.join(traindir, 'train')
        valdir = os.path.join(valdir, 'val')
    else:
        traindir = args.data[0]
        valdir= args.data[1]

    if(args.arch == "inception_v3"):
        raise RuntimeError("Currently, inception_v3 is not supported by this example.")
        # crop_size = 299
        # val_size = 320 # I chose this value arbitrarily, we can adjust.
    else:
        crop_size = 224
        val_size = 256


    print("dataloader=", args.dataloader)
    print("traindir =",  traindir)
    print("valdir =",  valdir)

    # tranverse all layers, and collect  layer num , init layer config info
    args.totalEpochs = args.epochs

    #setupModelQuantEnv(model,  args.qcfg , qname =args.arch ) 
    setupQuantEnv(args.qcfg, qname = args.arch)
    configModel( model)



    #args.totalLayerNumber =  getLayerCounter()
    logpath=QtArgs.logpath
    print("log path=",logpath)

    if  args.dataloader == "lmdb":
        print("[Info] Loading images from LMDB ", traindir)
        dataset = lmdbIter(args.batch_size, traindir)
        total= dataset.length
    
        pipelmdb = create_lmdb_pipeline(batch_size=args.batch_size,
                                num_threads=args.workers,
                                device_id=args.gpu,
                                seed=12 + args.local_rank,
                                dataset=dataset,
                                crop=crop_size,
                                size=val_size,
                                dali_cpu=args.dali_cpu,
                                shard_id=args.local_rank,
                                num_shards=args.world_size,
                                is_training=True)
        pipelmdb.build()
        train_lmdb_iter = LMDB_iter(pipelmdb, total)
        train_loader = iter(train_lmdb_iter)

        dataset = lmdbIter(args.batch_size, valdir)
        total= dataset.length
    
        pipelmdb = create_lmdb_pipeline(batch_size=args.batch_size,
                                num_threads=args.workers,
                                device_id=args.gpu,
                                seed=12 + args.local_rank,
                                dataset=dataset,
                                crop=crop_size,
                                size=val_size,
                                dali_cpu=args.dali_cpu,
                                shard_id=args.local_rank,
                                num_shards=args.world_size,
                                is_training=False)
        pipelmdb.build()
        val_lmdb_iter = LMDB_iter(pipelmdb, total)
        val_loader = iter(val_lmdb_iter)

    elif args.dataloader=="bing" :  # bing  
        print("[Info] Loading images from BING  files", traindir)
        trainbing  = BING(traindir, batch_size= args.batch_size, block_size = 1024, gpu=args.gpu)

        total = trainbing.length
        pipedali = create_lmdb_pipeline(batch_size=args.batch_size,
                                num_threads=args.workers,
                                device_id=args.gpu, # args.local_rank,
                                seed=12 + args.local_rank,
                                dataset= trainbing, #trainbing,
                                crop=crop_size,
                                size=val_size,
                                dali_cpu=args.dali_cpu,
                                shard_id=args.local_rank,
                                num_shards=args.world_size,
                                is_training=True)
                                    
        pipedali.build()
        train_lmdb_iter = LMDB_iter(pipedali, total)
        train_loader = iter(train_lmdb_iter)

        #train_loader = DALIClassificationIterator(pipedali, reader_name="Reader", last_batch_policy=LastBatchPolicy.PARTIAL)

        valbing  = BING(valdir, batch_size= args.batch_size , gpu=args.gpu)
        pipedali = create_lmdb_pipeline(batch_size=args.batch_size,
                                num_threads=args.workers,
                                device_id=args.gpu, #aargs.local_rank,
                                seed=12 + args.local_rank,
                                dataset= valbing, #  valbing,
                                crop=crop_size,
                                size=val_size,
                                dali_cpu=args.dali_cpu,
                                shard_id=args.local_rank,
                                num_shards=args.world_size,
                                is_training=False)
        pipedali.build()
        total = valbing.length
        val_lmdb_iter = LMDB_iter(pipedali, total)
        val_loader = iter(val_lmdb_iter)

        #val_loader = DALIClassificationIterator(pipedali, reader_name="Reader", last_batch_policy=LastBatchPolicy.PARTIAL)

        
    else :  # files 
        print("[Info] Loading images from .jpeg files", traindir)
        pipedali = create_dali_pipeline(batch_size=args.batch_size,
                                num_threads=args.workers,
                                device_id=args.gpu, # args.local_rank,
                                seed=12 + args.local_rank,
                                data_dir=traindir,
                                crop=crop_size,
                                size=val_size,
                                dali_cpu=args.dali_cpu,
                                shard_id=args.local_rank,
                                num_shards=args.world_size,
                                is_training=True)
                                    
        pipedali.build()
        train_loader = DALIClassificationIterator(pipedali, reader_name="Reader", last_batch_policy=LastBatchPolicy.PARTIAL)

        pipedali = create_dali_pipeline(batch_size=args.batch_size,
                                num_threads=args.workers,
                                device_id=args.gpu, #aargs.local_rank,
                                seed=12 + args.local_rank,
                                data_dir=valdir,
                                crop=crop_size,
                                size=val_size,
                                dali_cpu=args.dali_cpu,
                                shard_id=args.local_rank,
                                num_shards=args.world_size,
                                is_training=False)
        pipedali.build()
        val_loader = DALIClassificationIterator(pipedali, reader_name="Reader", last_batch_policy=LastBatchPolicy.PARTIAL)

    args.needActivationProfile = 0
    if  args.needActivationProfile:
        model.apply(add_activation_profile_hook)
        print("Activation  profling.........")
        top1, top5,loss= validate(train_loader,model, criterion,args, qConfig.profileDataPercent )
        print("Profiling finished .")
        print("Top1=%f  Top5=%f Loss=%f "%( top1, top5, loss))
        model.apply(remove_activation_profile_hook)



    if args.evaluate:
        print("-----------------Evaluating ---------------------------")
        model.eval() 

        #top1, top5, loss =validate(val_loader, model, criterion, args)
        if  args.qwm.lower() not in  ["ufp" ]:
            model.apply(quant_model_dfq)
            model.apply(add_hooks_for_eci_train)  
            top1, top5, loss =validate(val_loader, model, criterion, args)
            model.apply(remove_hooks_for_eci_train) 

            exit()

        # probing optimal weight E
        layersModel = getLayersModel()
        for l in range( len(layersModel)):
                layersModel[l].qwm = "none" 

        for l in range( len(layersModel)):
            best_top1 = 0. 
            best_WE =   1

            for e in [1,2,3,4,5]:            
                
                layersModel[l].qwm = "ufp" 
                layersModel[l].probingWE = e 

                model.apply(quant_model_dfq)
                model.apply(add_hooks_for_eci_train)  
                top1, top5, loss =validate(val_loader, model, criterion, args)
                model.apply(remove_hooks_for_eci_train) 
                val_loader.reset()
                if top1 > best_top1 :
                    best_top1 = top1
                    best_WE = e
                    print("    [INFO]     layer[%d] e=%d top1=%2.6f "%(l, e, top1 ), "best top1 : %2.6f   E=%d "%( best_top1, best_WE) )

            layersModel[l].optimalWE = best_WE  
            for t in range( l+1):
                print("[%d] : "%t, layersModel[t].optimalWE)

        
        print("\n\n  { ")
        for l in range( len(layersModel)):
               print(" %3.4f : %d ,   " %(layersModel[l].wts_lti,  layersModel[l].optimalWE))
        print(" ]")

        return   #top1,top5, loss

     
    # quantaztion aware training! QAT
    if args.qscheme == 'qat' :

        def qat_adjust_learning_rate(optimizer, epoch, args):
            """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
            lr = 0.00005
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        for epoch in range(args.start_epoch, args.epochs):
            if args.distributed:
                train_sampler.set_epoch(epoch)
            qat_adjust_learning_rate(optimizer, epoch, args)

            #train_quant_aware(train_loader, model, criterion, optimizer, epoch, args, needQ)
            train_qat(train_loader, model, criterion, optimizer, epoch, args )

            # evaluate on validation set
            acc1, acc5 , loss= validate(val_loader, model, criterion, args)

            # remember best acc@1 and save checkpoint
            is_best = acc1 > best_acc1
            #best_acc1 = max(acc1, best_acc1)


            if  epoch % 10  == 9 :
                fn = "./logs/"+args.arch+"/" + args.arch +"_"+ str( epoch+1) +".pth.tar"
                save_checkpoint({
                  'epoch': epoch + 1,
                  'arch': args.arch,
                  'state_dict': model.state_dict(),
                  'best_acc1': best_acc1,
                  'optimizer' : optimizer.state_dict(),
                  }, is_best, fn )

            if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                    and args.rank % ngpus_per_node == 0):
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'best_acc1': best_acc1,
                    'optimizer' : optimizer.state_dict(),
                }, is_best)
        return   acc1, acc5, loss

    ##  --FQT or Full Precision Training-------


    total_time = AverageMeter()

    if  args.FQT  >0 :
        print("[Info] Training with ECI8 ... args.FQT= ", args.FQT )

    train_start_time  = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        global  grad_lti_max, grad_lti_min;
        global  grad_E_max,  grad_E_min;
        global  E_distribution;

        setCurrentEpoch(epoch)  # by ECI

        _, used, free = env.getCudaMemInfo(args.gpu)
        print("\033[1;31m  [Epoch %d]Epoch Start :GPU memory used:  %6.2f  free: %6.2f \033[0m "%(epoch, used, free ))

        grad_lti_max =0.
        grad_lti_min =100. 
        grad_E_max  = 1
        grad_E_min  = 7
        E_distribution =[0] * 8

        #if args.FQT:
        model.apply(add_hooks_for_eci_train)  

        avg_train_time = train(train_loader, model, criterion, optimizer, epoch)

        if args.FQT>0 :  
            model.apply(update_model_profile )

        #model.apply(remove_hooks_for_eci_train) 

        total_time.update(avg_train_time)

        #if args.test:
        #    break

        _, used, free = env.getCudaMemInfo(args.gpu)
        print("\033[1;31m  [Epoch %d]Epoch Train:GPU memory used:  %6.2f  free: %6.2f \033[0m "%(epoch, used, free ))
        # evaluate on validation set
        [prec1, prec5, loss] =  validate(val_loader, model, criterion, args)
 
        acc1 = prec1
        acc5 = prec5

        QtStat.appendAccuracy(acc1, acc5, loss)

        #if args.FQT:
        model.apply(remove_hooks_for_eci_train) 

        _, used, free = env.getCudaMemInfo(args.gpu)
        print("\033[1;31m  [Epoch %d]Epoch Vlidation:GPU memory used:  %6.2f  free: %6.2f \033[0m "%(epoch, used, free ))
        
        #show_overall_profile(model)
 

        epoch_stat= {}
        epoch_stat['epoch'] = epoch
        epoch_stat['acc1'] = acc1
        epoch_stat['acc5'] = acc5
        epoch_stat['grad_lti_max'] = grad_lti_max
        epoch_stat['grad_lti_min'] = grad_lti_min
        epoch_stat['grad_E_max'] = grad_E_max
        epoch_stat['grad_E_min'] = grad_E_min
        epoch_stat['grad_magnitude'] = grad_magnitude
        epoch_stat['grad_E_distribution'] = grad_E_distribution
        
        epoch_stat['activation_lti_max'] = activation_lti_max
        epoch_stat['activation_lti_min'] = activation_lti_min
        epoch_stat['activation_E_max'] = activation_E_max
        epoch_stat['activation_E_min'] = activation_E_min
        epoch_stat['activation_magnitude'] = activation_magnitude
        epoch_stat['activation_E_distribution'] = activation_E_distribution

        curve.append(epoch_stat) 
 
        appendTrainCurveData("loss",loss)
        appendTrainCurveData("top1",acc1)
        recordTrainCurve("loss")
        recordTrainCurve("top1")

        recordCurve(logpath, args, curve, epoch , start_time)

       # recordMappingLTI2E(args.arch + "_LTI2E-"+ time.strftime('%Y%m%d_%H-%M',start_time)   + ".map")

        if args.local_rank == 0:
            is_best = prec1 > best_prec1
            if  epoch % 10  == 9 :
                #qm = "-W"+ args.qwm +'-A' + args.qam 
                #fn = "./logs/"+args.arch+  qm  +"/" + args.arch +"_"+ str( epoch+1) +".pth.tar"
                fn = logpath  +"/" + args.arch +"_"+ str( epoch) +".pth.tar"
                save_checkpoint({
                  'epoch': epoch + 1,
                  'arch': args.arch,
                  'state_dict': model.state_dict(),
                  'best_acc1': best_acc1,
                  'optimizer' : optimizer.state_dict(),
                  }, is_best, fn )


        # remember best prec@1 and save checkpoint
        if args.local_rank == 0:
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            save_checkpoint({
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer' : optimizer.state_dict(),
            }, is_best)
            if epoch == args.epochs - 1:
                print('##Top-1 {0}\n'
                      '##Top-5 {1}\n'
                      '##Perf  {2}'.format(
                      prec1,
                      prec5,
                      args.total_batch_size / total_time.avg))

        train_loader.reset()
        val_loader.reset()
        
        #show_overall_profile(model)
        recordQtStat(model, QtArgs)
        
    train_duration = time.time() - train_start_time


    #if args.FQT:
    model.apply(remove_hooks_for_eci_train)   

    recordQtStat(model, QtArgs)

    #show_overall_profile(model)

    print("[Info] Training Finished. Total duration: %f s. "%(train_duration) )

    recordCurve(logpath, args, curve, args.epochs, start_time)
    
    print(args)




def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.epoch = epoch
    model.train()
    end = time.time()
    t0 = time.time()
    for i, data in enumerate(train_loader):
        input = data[0]["data"]
        target = data[0]["label"].squeeze(-1).long()
        train_loader_len = int(math.ceil(train_loader._size / args.batch_size))


        if args.prof >= 0 and i == args.prof:
            print("Profiling begun at iteration {}".format(i))
            torch.cuda.cudart().cudaProfilerStart()

        if args.prof >= 0: torch.cuda.nvtx.range_push("Body of iteration {}".format(i))

        adjust_learning_rate(optimizer, epoch, i, train_loader_len)
        if args.test:
            if i > 400:
                break

        # compute output
        if args.prof >= 0: torch.cuda.nvtx.range_push("forward")
        output = model(input)
        if args.prof >= 0: torch.cuda.nvtx.range_pop()

        loss = criterion(output, target)

        # compute gradient and do SGD step
        optimizer.zero_grad()

        if args.prof >= 0: torch.cuda.nvtx.range_push("backward")
        if args.opt_level is not None:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
             loss.backward()

        if args.prof >= 0: torch.cuda.nvtx.range_pop()
 
        if args.FQT > 0 :
            model.apply(profile_gradient )

        if args.prof >= 0: torch.cuda.nvtx.range_push("optimizer.step()")
        optimizer.step()
        #if args.FQT:
        model.apply( backward_update_hp)  
        if args.prof >= 0: torch.cuda.nvtx.range_pop()

        if i%args.print_freq == args.print_freq -1:
            # Every print_freq iterations, check the loss, accuracy, and speed.
            # For best performance, it doesn't make sense to print these metrics every
            # iteration, since they incur an allreduce and some host<->device syncs.

            # Measure accuracy
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            
            # Average loss and accuracy across processes for logging
            if args.distributed:
                reduced_loss = reduce_tensor(loss.data)
                prec1 = reduce_tensor(prec1)
                prec5 = reduce_tensor(prec5)
            else:
                reduced_loss = loss.data

            # to_python_float incurs a host<->device sync
            losses.update(to_python_float(reduced_loss), input.size(0))
            top1.update(to_python_float(prec1), input.size(0))
            top5.update(to_python_float(prec5), input.size(0))

            torch.cuda.synchronize()
            batch_time.update((time.time() - end)/args.print_freq)
            end = time.time()

            if args.local_rank == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Speed {3:.3f} ({4:.3f})\t'
                      'Loss {loss.val:.10f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       epoch, i, train_loader_len,
                       args.world_size*args.batch_size/batch_time.val,
                       args.world_size*args.batch_size/batch_time.avg,
                       batch_time=batch_time,
                       loss=losses, top1=top1, top5=top5))

        # Pop range "Body of iteration {}".format(i)
        if args.prof >= 0: torch.cuda.nvtx.range_pop()

   
        if args.prof >= 0 and i == args.prof + 10:
            print("Profiling ended at iteration {}".format(i))
            torch.cuda.cudart().cudaProfilerStop()
            quit()
        if  args.test :
            if i>= 100:
                break
    return batch_time.avg



def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    end = time.time()


    for i, data in enumerate(val_loader):
        input = data[0]["data"]
        target = data[0]["label"].squeeze(-1).long()
        val_loader_len = int(val_loader._size / args.batch_size)

        # compute output
        with torch.no_grad():
            output = model(input)
            loss = criterion(output, target)
            
        outtgt= output.data.cpu()[0]
        
     
        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

        if args.distributed:
            reduced_loss = reduce_tensor(loss.data)
            prec1 = reduce_tensor(prec1)
            prec5 = reduce_tensor(prec5)
        else:
            reduced_loss = loss.data

        losses.update(to_python_float(reduced_loss), input.size(0))
        top1.update(to_python_float(prec1), input.size(0))
        top5.update(to_python_float(prec5), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # TODO:  Change timings to mirror train().

        if args.local_rank == 0 and i % (args.print_freq //4) == 0 :  # args.print_freq -1 :
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Speed {2:.3f} ({3:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, val_loader_len,
                   args.world_size * args.batch_size / batch_time.val,
                   args.world_size * args.batch_size / batch_time.avg,
                   batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))

        if args.test:
            if i >10:
                break

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
        .format(top1=top1, top5=top5))

    return [top1.avg, top5.avg, losses.avg]


def probe_validate(val_loader, model, criterion, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    """
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Probe: ')
   """

    # switch to evaluate mode
    model.eval()
    end = time.time()

    for i, data in enumerate(val_loader):
        input = data[0]["data"]
        target = data[0]["label"].squeeze(-1).long()
        val_loader_len = int(val_loader._size / args.batch_size)

        # compute output
        with torch.no_grad():
            output = model(input)
            loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

        if args.distributed:
            reduced_loss = reduce_tensor(loss.data)
            prec1 = reduce_tensor(prec1)
            prec5 = reduce_tensor(prec5)
        else:
            reduced_loss = loss.data

        losses.update(to_python_float(reduced_loss), input.size(0))
        top1.update(to_python_float(prec1), input.size(0))
        top5.update(to_python_float(prec5), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # TODO:  Change timings to mirror train().
        if args.local_rank == 0 and i % args.print_freq == args.print_freq -1 :
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Speed {2:.3f} ({3:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, val_loader_len,
                   args.world_size * args.batch_size / batch_time.val,
                   args.world_size * args.batch_size / batch_time.avg,
                   batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
        .format(top1=top1, top5=top5))

    return [top1.avg, top5.avg, losses.avg]



def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, step, len_epoch):
    """LR schedule that should yield 76% converged accuracy with batch size 256"""
    factor = epoch // 30

    if epoch >= 80:
        factor = factor + 1

    lr = args.lr*(0.1**factor)

    """Warmup"""
    if epoch < 5:
        lr = lr*float(1 + step + epoch*len_epoch)/(5.*len_epoch)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= args.world_size
    return rt




if __name__ == '__main__':

    main()
    summarizeQuantization()
    showenv()
    print("logpath=", QtArgs.logpath)
