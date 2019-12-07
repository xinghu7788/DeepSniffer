
#log_dir
log_dir="../../Results/Table4/logs/"
cd ../validate_deepsniffer

#vgg11
python 3_inference.py --model first_seg_ctc_v3 --restore_step 83 --sample_file ../dataset/typicalModels/vgg/vgg_sample11.log --label_file ../dataset/typicalModels/vgg/_klayer_11.log > vgg11.log

#vgg13
python 3_inference.py --model first_seg_ctc_v3 --restore_step 83 --sample_file ../dataset/typicalModels/vgg/vgg_sample13.log --label_file ../dataset/typicalModels/vgg/_klayer_13.log > vgg13.log

#vgg16
python 3_inference.py --model first_seg_ctc_v3 --restore_step 83 --sample_file ../dataset/typicalModels/vgg/vgg_sample16.log --label_file ../dataset/typicalModels/vgg/_klayer_16.log > vgg16.log

#vgg19
python 3_inference.py --model first_seg_ctc_v3 --restore_step 83 --sample_file ../dataset/typicalModels/vgg/vgg_sample19.log --label_file ../dataset/typicalModels/vgg/_klayer_19.log > vgg19.log

#resnet18
python 3_inference.py --model first_seg_ctc_v3 --restore_step 83 --sample_file ../dataset/typicalModels/resnet/resnet_sample18.log --label_file ../dataset/typicalModels/resnet/_klayer_18.log > resnet18.log

#resnet34
python 3_inference.py --model first_seg_ctc_v3 --restore_step 83 --sample_file ../dataset/typicalModels/resnet/resnet_sample34.log --label_file ../dataset/typicalModels/resnet/_klayer_34.log > resnet34.log

#resnet101
python 3_inference.py --model first_seg_ctc_v3 --restore_step 83 --sample_file ../dataset/typicalModels/resnet/resnet_sample101.log --label_file ../dataset/typicalModels/resnet/_klayer_101.log > resnet101.log

#resnet152
python 3_inference.py --model first_seg_ctc_v3 --restore_step 83 --sample_file ../dataset/typicalModels/resnet/resnet_sample152.log --label_file ../dataset/typicalModels/resnet/_klayer_152.log > resnet152.log


#nasnet
python 3_inference_dd_cutslice.py --model first_seg_ctc_v3_dd_norm_complete --restore_step 39 --sample_file ../dataset/typicalModels/nas_p_cutConcat/feats_nasnet_cutConcat_1.npy --label_file ../dataset/typicalModels/nas_p_cutConcat/klayer_nasnet_cutConcat_1.npy --seg_file ../dataset/typicalModels/nas_p_cutConcat/seg_nasnet_cutConcat_1.npy > nasnet.log

mv *.log ../../Results/Table4/logs/
