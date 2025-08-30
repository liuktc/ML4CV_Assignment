# python your_script.py \
#   --epochs 10 \
#   --lr 0.0001 \
#   --embedding_size 512 \
#   --pixel_per_class 50 \
#   --scale_factor 0.5 \
#   --preprocess resize \
#   --loss ProxyAnchor \
#   --optimizer AdamW \
#   --weight_decay 0.0001 \
#   --normalize_embeddings True \
#   --model DinoUpsampling \
#   --distance cos \
#   --loss_weighting fixed \
#   --lambda_metric 0.5 \
#   --lambda_ce 1.0 \
#   --save_path runs/dino_upsampling_proxyanchor.pth \
#   --plot_interval 500



python ./ML4CV_Assignment/run.py --save_path "proxy_metric_fixed.pth" --model DinoMetricLearning --embedding_size 256  --loss ProxyAnchor  --loss_weighting fixed --lambda_metric 0.1 --lambda_ce 1 --pixel_per_class 50 --kaggle
python ./ML4CV_Assignment/run.py --save_path "proxy_upsamlping_fixed.pth" --model DinoUpsampling --embedding_size 256  --loss ProxyAnchor  --loss_weighting fixed --lambda_metric 0.1 --lambda_ce 1 --pixel_per_class 50 --kaggle
python ./ML4CV_Assignment/run.py --save_path "triplet_metric_fixed.pth" --model DinoMetricLearning --embedding_size 256  --loss TripletMargin  --loss_weighting fixed --lambda_metric 0.1 --lambda_ce 1 --pixel_per_class 50 --kaggle
python ./ML4CV_Assignment/run.py --save_path "triplet_upsamlping_fixed.pth" --model DinoUpsampling --embedding_size 256  --loss TripletMargin  --loss_weighting fixed --lambda_metric 0.1 --lambda_ce 1 --pixel_per_class 50 --kaggle

