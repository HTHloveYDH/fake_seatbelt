python train.py --filters 32 64 128 256 512 --input_width 128 --input_height 128 --channel_num 3 --latent_dim 512 --loss_scale 1.0 --offset 0.0 --scale 255.0 --batch_size 32 --epochs 10 --channel_dim -1


python eval.py --load_image_path /data_ws/Data_1/tinghao/ae_fake/mand_3_norm_132.jpg --load_autoencoder_1_path /data_ws/Data_1/tinghao/ae_fake/saved_model/model/auto_encoder_1 --load_autoencoder_2_path /data_ws/Data_1/tinghao/ae_fake/saved_model/model/auto_encoder_2 --input_width 128 --input_height 128 --offset 0.0 --scale 255.0