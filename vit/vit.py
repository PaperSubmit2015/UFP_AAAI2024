# coding=utf-8
from __future__ import absolute_import, division, print_function

import logging
import argparse
import os
import sys
import random
import numpy as np

from datetime import timedelta

import torch
import torch.nn as nn
import torch.distributed as dist

from   torch.utils.dlpack import  to_dlpack, from_dlpack
from   torch.utils.cpp_extension import load



from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from apex import amp
from apex.parallel import DistributedDataParallel as DDP

from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from utils.data_utils import get_loader
from utils.dist_util import get_world_size


#from mx import finalize_mx_specs
#from mx import mx_mapping 
#from mx  import elemwise_ops, custom_extensions
mx_specs = {
    'w_elem_format': 'fp8_e4m3',
'a_elem_format': 'fp8_e4m3', 
'w_elem_format_bp' : 'fp8_e4m3', 

'a_elem_format_bp_ex': 'fp8_e4m3',
'a_elem_format_bp_os': 'fp8_e5m2' ,

 'block_size': 32,

'bfloat': 16,
'scale_bits'  :  8,
'custom_cuda': True, 
'quantize_backprop': True,
}

#final_mx_specs= finalize_mx_specs(mx_specs)

#mx_mapping.inject_pyt_ops(final_mx_specs)


from models.modeling import VisionTransformer, CONFIGS


curpath = os.path.dirname(os.path.realpath(__file__))
qpkgpath= os.path.abspath(os.path.join(curpath,os.pardir, "cnn")) 

sys.path.append(qpkgpath)


import qmethod  as qm
import qTensor  as qt
import plot
from   quant_model import *





logger = logging.getLogger(__name__)





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


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def save_model(args, model, epoch=0):
    model_to_save = model.module if hasattr(model, 'module') else model
    model_checkpoint = os.path.join(args.logpath, "%s_%d_checkpoint.bin" % (args.name, epoch))
    torch.save(model_to_save.state_dict(), model_checkpoint)
    logger.info("Saved model checkpoint to [DIR: %s]", args.output_dir)


def setup(args):
    # Prepare model
    config = CONFIGS[args.model_type]

    num_classes = 10 if args.dataset == "cifar10" else 100


    if args.dataset =="cifar10":
        num_classes = 10
    elif args.dataset =="cifar100":
        num_classes = 100
    else :
        num_classes = 1000
        args.traindir = "./imagenet/train"
        args.testdir = "./imagenet/val"
    


    model = VisionTransformer(config, args.img_size, zero_head=True, num_classes=num_classes)
    model.load_from(np.load(args.pretrained_dir))
    model.to(args.device)
    num_params = count_parameters(model)

    logger.info("{}".format(config))
    logger.info("Training parameters %s", args)
    logger.info("Total Parameter: \t%2.1fM" % num_params)
    print(num_params)
    return args, model


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def valid(args, model, writer, test_loader, global_step):
    # Validation!
    eval_losses = AverageMeter()

    logger.info("***** Running Validation *****")
    logger.info("  Num steps = %d", len(test_loader))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    all_preds, all_label = [], []
    epoch_iterator = tqdm(test_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True,
                          disable=args.local_rank not in [-1, 0])
    loss_fct = torch.nn.CrossEntropyLoss()
    for step, batch in enumerate(epoch_iterator):
        batch = tuple(t.to(args.device) for t in batch)
        x, y = batch
        with torch.no_grad():
            logits = model(x)[0]

            eval_loss = loss_fct(logits, y)
            eval_losses.update(eval_loss.item())

            preds = torch.argmax(logits, dim=-1)

        if len(all_preds) == 0:
            all_preds.append(preds.detach().cpu().numpy())
            all_label.append(y.detach().cpu().numpy())
        else:
            all_preds[0] = np.append(
                all_preds[0], preds.detach().cpu().numpy(), axis=0
            )
            all_label[0] = np.append(
                all_label[0], y.detach().cpu().numpy(), axis=0
            )
        epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)

    all_preds, all_label = all_preds[0], all_label[0]
    accuracy = simple_accuracy(all_preds, all_label)

    logger.info("\n")
    logger.info("Validation Results")
    logger.info("Global Steps: %d" % global_step)
    logger.info("Valid Loss: %2.5f" % eval_losses.avg)
    logger.info("Valid Accuracy: %2.5f" % accuracy)

    writer.add_scalar("test/accuracy", scalar_value=accuracy, global_step=global_step)
    return accuracy


def train(args, model):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir, exist_ok=True)
        #writer = SummaryWriter(log_dir=os.path.join("logs", args.name))
        writer = SummaryWriter(log_dir=args.logpath)

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    # Prepare dataset
    train_loader, test_loader = get_loader(args)

    # Prepare optimizer and scheduler
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.learning_rate,
                                momentum=0.9,
                                weight_decay=args.weight_decay)
    t_total = args.num_steps
    if args.decay_type == "cosine":
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    else:
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    if args.fp16:
        model, optimizer = amp.initialize(models=model,
                                          optimizers=optimizer,
                                          opt_level=args.fp16_opt_level)
        amp._amp_state.loss_scalers[0]._loss_scale = 2**20

    # Distributed training
    if args.local_rank != -1:
        model = DDP(model, message_size=250000000, gradient_predivide_factor=get_world_size())

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Total optimization steps = %d", args.num_steps)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)

    model.zero_grad()
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    losses = AverageMeter()
    global_step, best_acc = 0, 0

    currentEpoch=0

    # for debugging
    #model.eval()
    #accuracy = valid(args, model, writer, test_loader, global_step)
    #print("Accuracy =", accuracy)

    ##  ============
    setCurrentEpoch( global_step ) # added by ECI
    while True:
        print("[INFO] Epoch=%d"%(currentEpoch))           
        model.train()
        epoch_iterator = tqdm(train_loader,
                              desc="Training (X / X Steps) (loss=X.X)",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True,
                              disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            setCurrentEpoch(global_step)   ## added by ECI
            batch = tuple(t.to(args.device) for t in batch)
            x, y = batch
            loss = model(x, y)

            if global_step % 100 == 0 :
                appendTrainCurveData("loss",loss.cpu().item())
                recordTrainCurve(  "loss")

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                losses.update(loss.item()*args.gradient_accumulation_steps)
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scheduler.step()
                optimizer.step()
                
                model.apply( backward_update_hp)  

                optimizer.zero_grad()
                global_step += 1

                epoch_iterator.set_description(
                    "Training (%d / %d Steps) (loss=%2.5f) \n" % (global_step, t_total, losses.val)
                )
                if args.local_rank in [-1, 0]:
                    writer.add_scalar("train/loss", scalar_value=losses.val, global_step=global_step)
                    writer.add_scalar("train/lr", scalar_value=scheduler.get_lr()[0], global_step=global_step)
                if global_step % args.eval_every == 0 and args.local_rank in [-1, 0]:
                    accuracy = valid(args, model, writer, test_loader, global_step)
                    if best_acc < accuracy:
                        save_model(args, model, global_step)
                        best_acc = accuracy
                    appendTrainCurveData("top1", best_acc)   
                    recordTrainCurve(  "top1")
 
                    model.train()

                #show_overall_profile(model)
                if global_step % t_total == 0:
                    break
        losses.reset()
        if global_step % t_total == 0:
            break

 
    if args.local_rank in [-1, 0]:
        writer.close()
        
    logger.info("Best Accuracy: \t%f" % best_acc)
    logger.info("End Training!")


def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--name", required=True,
                        help="Name of this run. Used for monitoring.")
    parser.add_argument("--dataset", choices=["cifar10", "cifar100","imagenet"], default="cifar10",
                        help="Which downstream task.")
    parser.add_argument("--model_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16",
                                                 "ViT-L_32", "ViT-H_14", "R50-ViT-B_16"],
                        default="ViT-B_16",
                        help="Which variant to use.")
    parser.add_argument("--pretrained_dir", type=str, default="checkpoint/ViT-B_16.npz",
                        help="Where to search for pretrained ViT models.")
    parser.add_argument("--output_dir", default="output", type=str,
                        help="The output directory where checkpoints will be written.")

    parser.add_argument("--img_size", default=224, type=int,
                        help="Resolution size")
    parser.add_argument("--train_batch_size", default=512, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=64, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--eval_every", default=100, type=int,
                        help="Run prediction on validation set every so many steps."
                             "Will always run one evaluation at the end of training.")

    parser.add_argument("--learning_rate", default=3e-2, type=float,
                        help="The initial learning rate for SGD.")
    parser.add_argument("--weight_decay", default=0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--num_steps", default=10000, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine",
                        help="How to decay the learning rate.")
    parser.add_argument("--warmup_steps", default=500, type=int,
                        help="Step of training to perform learning rate warmup for.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O2',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")


    parser.add_argument('--qwm', default="none", type=str,
                     help='Apply quantization methods on weight or activation. '
                          'The quantization methods includes uq, eci, cdf,lloyd  ')
    parser.add_argument('--qwd', default=0, type=int,
                     help='Denote the quantization level in bits. If it is 0, it will search the optimal bit depth.'
                          '0 : using floating point.')
    parser.add_argument('--qwe', default=0, type=int,
                     help='Denote the E field of quantization number(EpMq,ECI). If it is 0, it will search the optimal bit depth.'
                          '0 : exploring the best E value .'   )
    parser.add_argument('--qwsr', default="self", type=str, 
                        help='Define  stocastic rounding mode  ') 

    parser.add_argument('--qam', default="none", type=str,
                     help='Apply quantization methods on weight or activation. '
                          'The quantization methods includes uq, eci, cdf,lloyd. Default is none(floating point). ')
    parser.add_argument('--qad', default=0, type=int,
                     help='Denote the quantization level in bits. If it is 0, it will search the optimal bit depth.'
                          '0 : using floating point.')
    parser.add_argument('--qae', default=2, type=int,
                     help='Denote the quantization level in bits. If it is 0, it will search the optimal bit depth.'
                          '0 : using floating point.')
    parser.add_argument('--qasr', default="self", type=str, 
                        help='Define  stocastic rounding mode  ') 

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

    parser.add_argument('--qL0', default=0, type=int,
                     help='Denote whether to quant the first layer(L0) . If it is 1, the first layer will be quantized.')
    parser.add_argument('--qLL', default=0, type=int,
                     help='Denote whether to quant the last layer(LL). If it is 1, the last layer(LL) will be quantized. ')

    parser.add_argument('--needProbeE', default=0, type=int,
                     help='Denote whether to quant the last FC layer. If it is 0, it will search the optimal bit depth.')

    parser.add_argument('--dataloader', type=str, default="file")
    parser.add_argument('-q', '--qscheme', metavar='QUANT', default='none', 
                    choices= ['none','float','ptq','qat'] ,
                    help=' Quantization scheme: none/float, ptq(post-training), qat(quantization aware) ')

    parser.add_argument('--statInterval', default=30, type=int,
                     help='Intervals stats. ')

    parser.add_argument('--verbose', default=0, type=int,    help='verbose. ')
    parser.add_argument('--logpath', default='./logs', type=str,
                               help='Direcotry  for storing log infomation.')
    parser.add_argument('--qconf', default="vit.json", type=str,    help='quantization configuration file ')

    args = parser.parse_args()
    args.arch= args.model_type
    args.totalEpochs = args.num_steps
    args.start_epoch = 0
 

    setupQuantEnv(args.qconf  )

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl',
                                             timeout=timedelta(minutes=60))
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s" %
                   (args.local_rank, args.device, args.n_gpu, bool(args.local_rank != -1), args.fp16))

    # Set seed
    set_seed(args)

    
    # Model & Tokenizer Setup
    args.arch= args.model_type
    args, model = setup(args)
    model.cpu()
    model.to(args.device)

    #setupModelQuantEnv(model, args.qconf )    
    configModel(model)    
    #QtArgs.logLevel =  (LOG_A_SNAPSHOT | LOG_W_SNAPSHOT | LOG_G_SNAPSHOT |  LOG_A_STAT |LOG_W_STAT |LOG_G_STAT |LOG_A_E |LOG_W_E |LOG_G_E   )
    QtArgs.logLayers = list(range(1000))
    #QtArgs.logLevel =  (LOG_A_SNAPSHOT | LOG_W_SNAPSHOT | LOG_G_SNAPSHOT)
    model.apply(add_hooks_for_eci_train)   # quant_activation



    # Training
    train(args, model)

    model.apply(remove_hooks_for_eci_train)  # quant_activation

if __name__ == "__main__":
    main()
    summarizeQuantization()
