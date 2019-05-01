#Citations to YunYang1994 
 
python scripts/extract_voc.py --voc_path ~/mmvc-ny-local/Munachiso_Nwadike/voc_dataset/VOCdevkit/train/ --dataset_info_path ./
cat ./2007_train.txt ./2007_val.txt > voc_train.txt
 
python scripts/extract_voc_test.py --voc_path ~/mmvc-ny-local/Munachiso_Nwadike/voc_dataset/VOCdevkit/test/ --dataset_info_path ./
cat ./2007_test.txt > voc_test.txt

python core/convert_tfrecord.py --dataset_txt ./voc_train.txt --tfrecord_path_prefix ~/mmvc-ny-local/Munachiso_Nwadike/voc_dataset/VOCdevkit/train/voc_train

python core/convert_tfrecord.py --dataset_txt ./voc_test.txt  --tfrecord_path_prefix ~/mmvc-ny-local/Munachiso_Nwadike/voc_dataset/VOCdevkit/test/voc_test
