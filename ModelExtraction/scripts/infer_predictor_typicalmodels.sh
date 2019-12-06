

#resnet18
python 3_inference.py --model first_seg_ctc_v3 --restore_step 83 --sample_file ../dataset/typicalModels/resnet/resnet_sample18.log --label_file ../dataset/typicalModels/resnet/_klayer_18.log > resnet.log

#resnet101
python 3_inference.py --model first_seg_ctc_v3 --restore_step 83 --sample_file ../dataset/typicalModels/resnet/resnet_sample101.log --label_file ../dataset/typicalModels/resnet/_klayer_101.log > resnet.log



#nasnet
python 3_inference_dd_cutslice.py --model first_seg_ctc_v3_dd_norm_complete --restore_step 39 --sample_file ../dataset/typicalModels/nas_p_cutConcat/feats_nasnet_cutConcat_1.npy --label_file ../dataset/typicalModels/nas_p_cutConcat/klayer_nasnet_cutConcat_1.npy --seg_file ../dataset/typicalModels/nas_p_cutConcat/seg_nasnet_cutConcat_1.npy > nasnet.log
