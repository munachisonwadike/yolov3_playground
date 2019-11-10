import argparse
import os
import numpy as np
import itertools
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataloaders import make_data_loader
from IPython.display import clear_output
from mypath import Path
from modeling.sync_batchnorm.replicate import patch_replication_callback
from modeling.deeplab import *
from skimage import io
from sklearn.metrics import confusion_matrix
from torch.autograd import Variable
from utils.loss import SegmentationLosses
from utils.calculate_weights import calculate_weigths_labels
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator
from visdom import Visdom
"""

\n
Notes:
1) Disabled code in dataloader/make_data_loader/__init__.py
 that used SBD since it simply was preventing the code from running

2) Set the code in GPU ids that allowed to use multiple GPU's- 
double checkt that it does use all GPUs later

3) Manually added the context encoding module to the decoder to create munanet
Still not sure what the 'lateral' argument from encnets context encoder does
so mess with it as True and False in training to see if it improves performance
Also not sure why the original encnet paper always casts the return value
of a forward function to a tuple and casts back to a list (tensor) when using it
so just left it that way for now. Gonna trying putting the context encoder
at different stages of the network such as decoder, encoder, and different stages
within these. Also ignored the semantic encoding loss since you can't really include it
at the encoder stage. Perhaps figure out a way to add a semantic encoding loss later on 
as Xiang discussed 

4) even when manually adding context encoding modules, the function syncbacth 
norm uses c++ and cuda files so you still need to copy the encoding/lib folder 
from the PyTorchEncoding repo note that when you import lib from /encoding, you 
may need to remount the disk you are working on so that enclibcpu can import properly 
https://stackoverflow.com/questions/13502156/what-are-possible-causes-of-failed-to-map-segment-from-shared-object-operation
checking out /etc/fstab showed me that I needed to remount ~ .../ mmvc-ad-local-002
also note that you need to run the setup.py for the encoding/lib/cpu and encoding/lib/gpu 
to install and to run build_ext - this will require getting all the correct pytorch,
 cuda versions (perhaps pytorch nightly)
and even gcc

5) messed with the number of epochs and used early stopping

6) set the learning rate to .01. Perhaps change this? 

\n
"""

#bro import lib from decoder!
class Trainer(object):
    palette = {0 : (255, 255, 255), # Impervious surfaces (white)
           1 : (0, 0, 255),     # Buildings (blue)
           2 : (0, 255, 255),   # Low vegetation (cyan)
           3 : (0, 255, 0),     # Trees (green)
           4 : (255, 255, 0),   # Cars (yellow)
           5 : (255, 0, 0),     # Clutter (red)
           6 : (0, 0, 0)}       # Undefined (black)

    invert_palette = {v: k for k, v in palette.items()}

    WINDOW_SIZE = (256, 256) # Patch size
    STRIDE = 32 # Stride for testing
    IN_CHANNELS = 3 # Number of input channels (e.g. RGB)
    FOLDER = "/home/mmvc/mmvc-ny-local/Munachiso_Nwadike/ISPRS_DATASET/" # Replace with your "/path/to/the/ISPRS/dataset/folder/"
    BATCH_SIZE = 4 # Number of samples in a mini-batch

    LABELS = ["roads", "buildings", "low veg.", "trees", "cars", "clutter"] # Label names
    N_CLASSES = len(LABELS) # Number of classes
    WEIGHTS = torch.ones(N_CLASSES) # Weights for class balancing
    CACHE = True # Store the dataset in-memory

    DATASET = 'Vaihingen'

    if DATASET == 'Potsdam':
        MAIN_FOLDER = FOLDER + 'Potsdam/'
        DATA_FOLDER = MAIN_FOLDER + '3_Ortho_IRRG/top_potsdam_{}_{}_IRRG.tif'
        LABEL_FOLDER = MAIN_FOLDER + '5_Labels_for_participants/top_potsdam_{}_{}_label.tif'
        ERODED_FOLDER = MAIN_FOLDER + '5_Labels_for_participants_no_Boundary/top_potsdam_{}_label_noBoundary.tif'    
    elif DATASET == 'Vaihingen':
        MAIN_FOLDER = FOLDER + 'Vaihingen/'
        DATA_FOLDER = MAIN_FOLDER + 'top/top_mosaic_09cm_area{}.tif'
        LABEL_FOLDER = MAIN_FOLDER + 'gts_for_participants/top_mosaic_09cm_area{}.tif'
        ERODED_FOLDER = MAIN_FOLDER + 'gts_eroded_complete/top_mosaic_09cm_area{}.tif'


    def __init__(self, args):
        self.args = args

        # Define Saver
        self.saver = Saver(args)
        self.saver.save_experiment_config()
        # Define Tensorboard Summary
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()
        
        # Define Dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        self.train_loader, self.val_loader, self.test_ids, self.nclass = make_data_loader(self.ERODED_FOLDER, args, **kwargs)

        # Define network
        model = DeepLab(num_classes=self.nclass,
                        backbone=args.backbone,
                        output_stride=args.out_stride,
                        sync_bn=args.sync_bn,
                        freeze_bn=args.freeze_bn)

        train_params = [{'params': model.get_1x_lr_params(), 'lr': args.lr},
                        {'params': model.get_10x_lr_params(), 'lr': args.lr * 10}]

        # Define Optimizer
        optimizer = torch.optim.SGD(train_params, momentum=args.momentum,
                                    weight_decay=args.weight_decay, nesterov=args.nesterov)

        # Define Criterion
        # whether to use class balanced weights
        if args.use_balanced_weights:
            classes_weights_path = os.path.join(Path.db_root_dir(args.dataset), args.dataset+'_classes_weights.npy')
            if os.path.isfile(classes_weights_path):
                weight = np.load(classes_weights_path)
            else:
                weight = calculate_weigths_labels(args.dataset, self.train_loader, self.nclass)
            weight = torch.from_numpy(weight.astype(np.float32))
        else:
            weight = None
        self.criterion = SegmentationLosses(weight=weight, cuda=args.cuda).build_loss(mode=args.loss_type)
        self.model, self.optimizer = model, optimizer
        
        # Define Evaluator
        self.evaluator = Evaluator(self.nclass)
        # Define lr scheduler
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr,
                                            args.epochs, len(self.train_loader))

        # Using cuda
        if args.cuda:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            patch_replication_callback(self.model)
            self.model = self.model.cuda()

        # Resuming checkpoint
        self.best_pred = 0.0
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            if args.cuda:
                self.model.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            if not args.ft:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))

        # Clear start epoch if fine-tuning
        if args.ft:
            args.start_epoch = 0

        self.plotter = VisdomLinePlotter(env_name='Plots')

    def training(self, epoch):
        train_loss = 0.0
        self.model.train()
        tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()
            output = self.model(image)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))
            self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)
            
        self.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print('Loss: %.3f' % train_loss)

        if self.args.no_val:
            # save checkpoint every epoch
            is_best = False
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)
        #plot loss values every epoch
        self.plotter.plot('loss', 'train', 'Train Loss b4lr004 (with contec)', epoch, train_loss)

    def validation(self, epoch):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc='\r')
        test_loss = 0.0
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output = self.model(image)
            loss = self.criterion(output, target)
            test_loss += loss.item()
            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            # Add batch sample into evaluator
            self.evaluator.add_batch(target, pred)

        # Fast test during the training
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        self.writer.add_scalar('val/total_loss_epoch', test_loss, epoch)
        self.writer.add_scalar('val/mIoU', mIoU, epoch)
        self.writer.add_scalar('val/Acc', Acc, epoch)
        self.writer.add_scalar('val/Acc_class', Acc_class, epoch)
        self.writer.add_scalar('val/fwIoU', FWIoU, epoch)
        print('Validation:')
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
        print('Loss: %.3f' % test_loss)

        new_pred = mIoU
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)

        #plot needed testing values 
        self.plotter.plot('loss', 'val', 'Validation Loss b4lr004', epoch, test_loss)
        self.plotter.plot('acc', 'val', 'Validation Accuracy b4lr004', epoch, Acc)
        self.plotter.plot('acc_class', 'MNET val', 'Class Accuracy b4lr004', epoch, Acc_class)
        self.plotter.plot('mIoU', 'val', 'Validation Mean IoU b4lr004', epoch, mIoU) 
        self.plotter.plot('fwIoU', 'val', 'Validation fwIoU b4lr004', epoch, FWIoU) 

    def convert_to_color(self, arr_2d):
        """ Numeric labels to RGB-color encoding """
        palette=self.palette

        arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)

        for c, i in palette.items():
            m = arr_2d == c
            arr_3d[m] = i

        return arr_3d

    def convert_from_color(self, arr_3d):
        """ RGB-color encoding to grayscale labels """
        palette=self.invert_palette

        arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)

        for c, i in palette.items():
            m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
            arr_2d[m] = i

        return arr_2d


    def sliding_window(self, top, step, window_size):
        """ Slide a window_shape window across the image with a stride of step """
        top = np.array(top)
        for x in range(0, top.shape[0], step):
            if x + window_size[0] > top.shape[0]:
                x = top.shape[0] - window_size[0]
            for y in range(0, top.shape[1], step):
                if y + window_size[1] > top.shape[1]:
                    y = top.shape[1] - window_size[1]
                yield x, y, window_size[0], window_size[1]
            
    def count_sliding_window(self, top, step, window_size):
        """ Count the number of windows in an image """
        top = np.array(top)
        c = 0
        for x in range(0, top.shape[0], step):
            if x + window_size[0] > top.shape[0]:
                x = top.shape[0] - window_size[0]
            for y in range(0, top.shape[1], step):
                if y + window_size[1] > top.shape[1]:
                    y = top.shape[1] - window_size[1]
                c += 1
        return c

    def grouper(self, n, iterable):
        """ Browse an iterator by chunk of n elements """
        it = iter(iterable)
        while True:
            chunk = tuple(itertools.islice(it, n))
            if not chunk:
                return
            yield chunk

    def metrics(self, epoch, predictions, gts):
        label_values=self.LABELS

        cm = confusion_matrix(
                gts,
                predictions,
                range(len(label_values)))
        
        print("Confusion matrix :")
        print(cm)
        
        print("---")
        
        # Compute global accuracy
        total = sum(sum(cm))
        accuracy = sum([cm[x][x] for x in range(len(cm))])
        accuracy *= 100 / float(total)
        print("{} pixels processed".format(total))
        print("Total accuracy : {}%".format(accuracy))
        
        print("---")
        
        # Compute F1 score
        F1Score = np.zeros(len(label_values))
        for i in range(len(label_values)):
            try:
                F1Score[i] = 2. * cm[i,i] / (np.sum(cm[i,:]) + np.sum(cm[:,i]))
            except:
                # Ignore exception if there is no element in class i for test set
                pass
        print("F1Score :")
        for l_id, score in enumerate(F1Score):
            print("{}: {}".format(label_values[l_id], score))

            self.plotter.plot(label_values[l_id], 'test', 'F1Score b4lr004' + str(label_values[l_id]), epoch, score)

        print("---")
            
        # Compute kappa coefficient
        total = np.sum(cm)
        pa = np.trace(cm) / float(total)
        pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / float(total*total)
        kappa = (pa - pe) / (1 - pe);
        print("Kappa: " + str(kappa))
        return accuracy

    def test(self, test_ids, epoch, all=False):
    # Use the network on the test set
        strides=self.WINDOW_SIZE[0]
        batch_size=self.BATCH_SIZE
        window_size=self.WINDOW_SIZE
        test_images = (1 / 255 * np.asarray(io.imread(self.DATA_FOLDER.format(id)), dtype='float32') for id in test_ids)
        test_labels = (np.asarray(io.imread(self.LABEL_FOLDER.format(id)), dtype='uint8') for id in test_ids)
        eroded_labels = (self.convert_from_color(io.imread(self.ERODED_FOLDER.format(id))) for id in test_ids)
        all_preds = []
        all_gts = []
        
        # Switch the network to inference mode
        self.model.eval()

        for img, gt, gt_e in tqdm(zip(test_images, test_labels, eroded_labels), total=len(test_ids), leave=False):
            pred = np.zeros(img.shape[:2] + (self.N_CLASSES,))

            total = self.count_sliding_window(img, strides, window_size) // batch_size
            for i, coords in enumerate(tqdm(self.grouper(batch_size, self.sliding_window(img, strides, window_size)), total=total, leave=False)):
                
                # Build the tensor
                image_patches = [np.copy(img[x:x+w, y:y+h]).transpose((2,0,1)) for x,y,w,h in coords]
                image_patches = np.asarray(image_patches)
                image_patches = Variable(torch.from_numpy(image_patches).cuda(), volatile=True)
                
                # Do the inference
                outs = self.model(image_patches)
                outs = outs.data.cpu().numpy()
                
                # Fill in the results array
                for out, (x, y, w, h) in zip(outs, coords):
                    out = out.transpose((1,2,0))
                    pred[x:x+w, y:y+h] += out
                del(outs)

            pred = np.argmax(pred, axis=-1)

            # Display the result
            # clear_output()
            # fig = plt.figure()
            # plt1 = fig.add_subplot(1,3,1)
            # plt.imshow(np.asarray(255 * img, dtype='uint8'))
            # plt2 = fig.add_subplot(1,3,2)
            # plt.imshow(self.convert_to_color(pred))
            # plt3 = fig.add_subplot(1,3,3)
            # plt.imshow(gt)

            # plt1.set_title('RGB')
            # plt2.set_title('Predictions')
            # plt3.set_title('Ground Truth')

            # if epoch == 70:
            #     plt.draw()
            #     plt.pause(.0001)

            all_preds.append(pred)
            all_gts.append(gt_e)

            clear_output()
            # Compute some metrics
            self.metrics(epoch, pred.ravel(), gt_e.ravel())
            accuracy = self.metrics(epoch, np.concatenate([p.ravel() for p in all_preds]), np.concatenate([p.ravel() for p in all_gts]).ravel())
        self.plotter.plot('accuracy', 'test', 'Accuracy b4lr004 (with contec)', epoch, accuracy) 
        if all:
            return accuracy, all_preds, all_gts
        else:
            return accuracy

class VisdomLinePlotter(object):
    """Plots to Visdom"""
    def __init__(self, env_name='main'):
        self.viz = Visdom()
        self.env = env_name
        self.plots = {}
    def plot(self, var_name, split_name, title_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
                legend=[split_name],
                title=title_name,
                xlabel='Epochs',
                ylabel=var_name
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name, update = 'append')

def main():

    print("""
       SEE NOTES IN COMMENTS
            """)
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
    parser.add_argument('--backbone', type=str, default='resnet',
                        choices=['resnet', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--out-stride', type=int, default=16,
                        help='network output stride (default: 8)')
    parser.add_argument('--dataset', type=str, default='vaihingen',
                        choices=['vaihingen', 'potsdam', 'pascal', 'coco', 'cityscapes'],
                        help='dataset name (default: pascal)')
    parser.add_argument('--use-sbd', action='store_true', default=True,
                        help='whether to use SBD dataset (default: True)')
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--base-size', type=int, default=513,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=513,
                        help='crop image size')
    parser.add_argument('--sync-bn', type=bool, default=None,
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')
    parser.add_argument('--loss-type', type=str, default='ce',
                        choices=['ce', 'focal'],
                        help='loss func type (default: ce)')
    # training hyper params
    parser.add_argument('--epochs', type=int, default=None, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                training (default: auto)')
    parser.add_argument('--test-batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                testing (default: auto)')
    parser.add_argument('--use-balanced-weights', action='store_true', default=False,
                        help='whether to use balanced weights (default: False)')
    # optimizer params
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (default: auto)')
    parser.add_argument('--lr-scheduler', type=str, default='poly',
                        choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode: (default: poly)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--nesterov', action='store_true', default=False,
                        help='whether use nesterov (default: False)')
    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true', default=
                        False, help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default=None,
                        help='set the checkpoint name')
    # finetuning pre-trained models
    parser.add_argument('--ft', action='store_true', default=False,
                        help='finetuning on a different dataset')
    # evaluation option
    parser.add_argument('--eval-interval', type=int, default=1,
                        help='evaluuation interval (default: 1)')
    parser.add_argument('--no-val', action='store_true', default=False,
                        help='skip validation during training')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            # args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
            args.gpu_ids = [0]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    if args.sync_bn is None:
        if args.cuda and len(args.gpu_ids) > 1:
            args.sync_bn = True
        else:
            args.sync_bn = False

    # default settings for epochs, batch_size and lr
    if args.epochs is None:
        epoches = {
            'coco': 30,
            'cityscapes': 200,
            'pascal': 5000,  #formerly 50 but I train till convergence
            'vaihingen': 1000 #only need 52 epochs for learning rate of 0.001, ... for 0.002
        }
        args.epochs = epoches[args.dataset.lower()]

    if args.batch_size is None:
        args.batch_size = 4 * len(args.gpu_ids)

    if args.test_batch_size is None:
        args.test_batch_size = args.batch_size

    if args.lr is None:
        lrs = {
            'coco': 0.1,
            'cityscapes': 0.01,
            'pascal': 0.007,
            'vaihingen': 0.004 #note for next time to try b=4/5 lr=.001 which touched 91%/90%, #perhaps change this?
        }
        args.lr = lrs[args.dataset.lower()] / (4 * len(args.gpu_ids)) * args.batch_size


    if args.checkname is None:
        args.checkname = 'deeplab-'+str(args.backbone)
    print(args)
    torch.manual_seed(args.seed)
    trainer = Trainer(args)
    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epoches:', trainer.args.epochs)
    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        trainer.training(epoch)
        if not trainer.args.no_val and epoch % args.eval_interval == (args.eval_interval - 1):
            trainer.validation(epoch)
        trainer.test(trainer.test_ids, epoch)

    trainer.writer.close()

if __name__ == "__main__":
   main()
