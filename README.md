# OncoNet: Developing Deep Learning Models for Mammography

# Introduction
This repository was used to develop the models described in:

- [A Deep Learning Mammography-Based Model for Improved Breast Cancer Risk Prediction](https://pubs.rsna.org/doi/)
- [Mammographic Breast Density Assessment Using
Deep Learning: Clinical Implementation](https://pubs.rsna.org/doi/10.1148/radiol.2018180694)

As the repository is updated with new papers, we will create a new branch for each.

# Usage
This code base is provided for clarify implementation details. It is not runnable in a stand-alone fashion since it assumes access to images/metafiles to initialize the Dataset object. If you want to try the models described in the papers above, please checkout [OncoServe](https://github.com/yala/OncoServe_Public), our model deployment codebase. It supports running all of our published models with a webserver HTTP interface.

# Command to train the Density Model:
```
CUDA_VISIBLE_DEVICES=0 python -u scripts/main.py  --batch_size 32 --cuda --dataset mgh_mammo_full_density --dropout 0.4 --epochs 100 --img_dir /scratch1/mammosprint --img_mean 7662.576866173061 --img_std 12594.148555576781 --img_size 256 256 --init_lr 0.0001 --cluster_exams --metadata_dir /home/administrator/Mounts/Isilon/metadata --model_name resnet18 --num_chan 3 --num_workers 20 --objective cross_entropy --optimizer adam --patience 10 --pretrained_on_imagenet --run_prefix snapshot --save_dir snapshot --train --test --image_transformers scale_2d rand_hor_flip rand_ver_flip rotate_90 rotate_range/max=10/min=-10 --tensor_transformers force_num_chan_2d normalize_2d
```

# Command to train the ImageOnly DL (MIRAI v0.1)
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -u scripts/main.py  --batch_size 24 --batch_splits 2 --cuda --dataset mgh_mammo_5year_risk --cluster_exams --weight_decay 5e-05 --momentum 0.9 --epochs 15 --lr_decay 0.1 --img_dir /home/administrator/Mounts/pngs16 --train_years 2012 2011 2010 2009 --dev_years 2012 2011 2010 2009 --test_years 2012 2011 2010 2009 --img_mean 7047.99 --img_size 1664 2048 --img_std 12005.5 --init_lr 0.0001 --metadata_dir /home/administrator/Mounts/Isilon/metadata --model_name custom_resnet --pool_name GlobalMaxPool --block_layout BasicBlock,2 BasicBlock,2 BasicBlock,2 BasicBlock,2 --dropout 0 --num_chan 3 --tuning_metric auc --num_workers 24 --objective cross_entropy --optimizer adam --patience 10 --max_batches_per_train_epoch 1500 --max_batches_per_dev_epoch 1500 --pretrained_on_imagenet --run_prefix snapshot --save_dir snapshot/ --train --test --class_bal --image_transformers scale_2d align_to_left rand_ver_flip rotate_range/min=-20/max=20 --tensor_transformers force_num_chan_2d normalize_2d --test_image_transformers scale_2d align_to_left --test_tensor_transformers force_num_chan_2d normalize_2d --data_parallel --num_gpus 4 --num_shards 1
```

# Command to train the HybridDL (MIRAI v0.2)
```
CUDA_VISIBLE_DEVICES=4,5,6 python -u scripts/main.py  --batch_size 24 --batch_splits 2 --cuda --dataset mgh_mammo_5year_risk --cluster_exams --weight_decay 5e-05 --momentum 0.9 --epochs 15 --lr_decay 0.1 --img_dir /home/administrator/Mounts/pngs16 --train_years 2012 2011 2010 2009 --dev_years 2012 2011 2010 2009 --test_years 2012 2011 2010 2009 --img_mean 7047.99 --img_size 1664 2048 --img_std 12005.5 --init_lr 0.0001 --metadata_dir /home/administrator/Mounts/Isilon/metadata --model_name custom_resnet --pool_name GlobalMaxPool --block_layout BasicBlock,2 BasicBlock,2 BasicBlock,2 BasicBlock,2 --num_chan 3 --tuning_metric auc --num_workers 24 --objective cross_entropy --optimizer adam --patience 10 --max_batches_per_train_epoch 1500 --max_batches_per_dev_epoch 1500 --pretrained_on_imagenet --run_prefix snapshot --save_dir snapshot/ --train --test --class_bal --image_transformers scale_2d align_to_left rand_ver_flip rotate_range/min=-20/max=20 --tensor_transformers force_num_chan_2d normalize_2d --test_image_transformers scale_2d align_to_left --test_tensor_transformers force_num_chan_2d normalize_2d --data_parallel --num_gpus 3 --num_shards 1 --use_risk_factors --dropout 0 --risk_factor_metadata_path /home/administrator/Mounts/Isilon/metadata/risk_factors_aug06_2018_mammo_and_mri.json --risk_factor_keys density binary_family_history binary_biopsy_benign binary_biopsy_LCIS binary_biopsy_atypical_hyperplasia age menarche_age menopause_age first_pregnancy_age prior_hist race parous menopausal_status weight height ovarian_cancer ovarian_cancer_age ashkenazi brca mom_bc_cancer_history m_aunt_bc_cancer_history p_aunt_bc_cancer_history m_grandmother_bc_cancer_history p_grantmother_bc_cancer_history sister_bc_cancer_history mom_oc_cancer_history m_aunt_oc_cancer_history p_aunt_oc_cancer_history m_grandmother_oc_cancer_history p_grantmother_oc_cancer_history sister_oc_cancer_history hrt_type hrt_duration hrt_years_ago_stopped
```
