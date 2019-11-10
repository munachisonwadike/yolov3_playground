from glob import glob
import random
from dataloaders.datasets import cityscapes, coco, combine_dbs, pascal, sbd, vaihingen
from torch.utils.data import DataLoader

def make_data_loader(ERODED_FOLDER, args, **kwargs): 
    #the numbers in vaihingen aren't sequential- 9, 18, 19, 25, and 36 are missing so just 33 images

    if args.dataset == 'vaihingen' or args.dataset == 'potsdam':
        

        # Load the datasets
        if args.dataset == 'potsdam':
            all_files = sorted(glob(ERODED_FOLDER.replace('{}', '*')))
            all_ids = ["_".join(f.split('_')[3:5]) for f in all_files]
        elif args.dataset == 'vaihingen':
            #all_ids = 
            all_files = sorted(glob(ERODED_FOLDER.replace('{}', '*')))
            all_ids = [f.split('area')[-1].split('.')[0].split('_')[0] for f in all_files]
        # Random tile numbers for train/test split
        # train_ids = random.sample(all_ids, 2 * len(all_ids) // 3 + 1)
        # test_ids = list(set(all_ids) - set(train_ids))
 
        #for the full, final run
        train_ids = [1, 3, 5, 7, 11, 13, 15, 17, 21, 23, 26, 28, 30, 32, 34, 37]
        test_ids = [2, 4, 6, 8, 10, 12, 14, 16, 20, 22, 24, 27, 29, 31, 33, 35, 38]

         #for the one-fold cross validation to pick hyperperameters
        # train_ids = [1, 3, 23, 26, 7, 11, 13, 28, 17, 32, 34]
        # test_ids = [5, 21, 15, 30, 37]
        
        print("\n Tiles for training : ", train_ids)
        print("Tiles for testing : ", test_ids, "\n")


        CACHE = True # Store the dataset in-memory
        train_set = vaihingen.VaihingenSegmentation(train_ids, cache=CACHE)
        val_set = vaihingen.VaihingenSegmentation(test_ids, cache=CACHE)

        if args.use_sbd:
            pass
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True, **kwargs)

        return train_loader, val_loader, test_ids, num_class

        
    elif args.dataset == 'pascal':
        train_set = pascal.VOCSegmentation(args, split='train')
        val_set = pascal.VOCSegmentation(args, split='val')
        if args.use_sbd:
            pass
        #     sbd_train = sbd.SBDSegmentation(args, split=['train', 'val'])
        #     train_set = combine_dbs.CombineDBs([train_set, sbd_train], excluded=[val_set])

        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = None

        return train_loader, val_loader, test_loader, num_class

    elif args.dataset == 'cityscapes':
        train_set = cityscapes.CityscapesSegmentation(args, split='train')
        val_set = cityscapes.CityscapesSegmentation(args, split='val')
        test_set = cityscapes.CityscapesSegmentation(args, split='test')
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)

        return train_loader, val_loader, test_loader, num_class

    elif args.dataset == 'coco':
        train_set = coco.COCOSegmentation(args, split='train')
        val_set = coco.COCOSegmentation(args, split='val')
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = None
        return train_loader, val_loader, test_loader, num_class

    else:
        raise NotImplementedError

