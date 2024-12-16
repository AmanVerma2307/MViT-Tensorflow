#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python './Scripts/MS_MViT_SOLI_Trainer.py' --lambda_id 0.5 --lambda_cgid 0.5 --exp_name MS_MViT_pt5-pt5_SOLI
CUDA_VISIBLE_DEVICES=0 python './Scripts/MS_MViT_SOLI_Tester.py' --lambda_id 0.5 --lambda_cgid 0.5 --exp_name MS_MViT_pt5-pt5_SOLI

CUDA_VISIBLE_DEVICES=0 python './Scripts/MS_MViT_SOLI_Trainer.py' --lambda_id 0.5 --lambda_cgid 1.0 --exp_name MS_MViT_pt5-1_SOLI
CUDA_VISIBLE_DEVICES=0 python './Scripts/MS_MViT_SOLI_Tester.py' --lambda_id 0.5 --lambda_cgid 1.0 --exp_name MS_MViT_pt5-1_SOLI

CUDA_VISIBLE_DEVICES=0 python './Scripts/MS_MViT_SOLI_Trainer.py' --lambda_id 0.5 --lambda_cgid 1.5 --exp_name MS_MViT_pt5-1pt5_SOLI
CUDA_VISIBLE_DEVICES=0 python './Scripts/MS_MViT_SOLI_Tester.py' --lambda_id 0.5 --lambda_cgid 1.5 --exp_name MS_MViT_pt5-1pt5_SOLI

CUDA_VISIBLE_DEVICES=0 python './Scripts/MS_MViT_SOLI_Trainer.py' --lambda_id 1.0 --lambda_cgid 0.5 --exp_name MS_MViT_1-pt5_SOLI
CUDA_VISIBLE_DEVICES=0 python './Scripts/MS_MViT_SOLI_Tester.py' --lambda_id 1.0 --lambda_cgid 0.5 --exp_name MS_MViT_1-pt5_SOLI

CUDA_VISIBLE_DEVICES=0 python './Scripts/MS_MViT_SOLI_Trainer.py' --lambda_id 1.0 --lambda_cgid 1.0 --exp_name MS_MViT_1-1_SOLI
CUDA_VISIBLE_DEVICES=0 python './Scripts/MS_MViT_SOLI_Tester.py' --lambda_id 1.0 --lambda_cgid 1.0 --exp_name MS_MViT_1-1_SOLI

CUDA_VISIBLE_DEVICES=0 python './Scripts/MS_MViT_SOLI_Trainer.py' --lambda_id 1.0 --lambda_cgid 1.5 --exp_name MS_MViT_1-1pt5_SOLI
CUDA_VISIBLE_DEVICES=0 python './Scripts/MS_MViT_SOLI_Tester.py' --lambda_id 1.0 --lambda_cgid 1.5 --exp_name MS_MViT_1-1pt5_SOLI

CUDA_VISIBLE_DEVICES=0 python './Scripts/MS_MViT_SOLI_Trainer.py' --lambda_id 1.5 --lambda_cgid 0.5 --exp_name MS_MViT_1pt5-pt5_SOLI
CUDA_VISIBLE_DEVICES=0 python './Scripts/MS_MViT_SOLI_Tester.py' --lambda_id 1.5 --lambda_cgid 0.5 --exp_name MS_MViT_1pt5-pt5_SOLI

CUDA_VISIBLE_DEVICES=0 python './Scripts/MS_MViT_SOLI_Trainer.py' --lambda_id 1.5 --lambda_cgid 1.0 --exp_name MS_MViT_1pt5-1_SOLI
CUDA_VISIBLE_DEVICES=0 python './Scripts/MS_MViT_SOLI_Tester.py' --lambda_id 1.5 --lambda_cgid 1.0 --exp_name MS_MViT_1pt5-1_SOLI

CUDA_VISIBLE_DEVICES=0 python './Scripts/MS_MViT_SOLI_Trainer.py' --lambda_id 1.5 --lambda_cgid 1.5 --exp_name MS_MViT_1pt5-1pt5_SOLI
CUDA_VISIBLE_DEVICES=0 python './Scripts/MS_MViT_SOLI_Tester.py' --lambda_id 1.5 --lambda_cgid 1.5 --exp_name MS_MViT_1pt5-1pt5_SOLI