
for ratio in 0.1 0.3 0.5 0.7
do
for model in 'PSW-I'   
do
if [ $model = 'PSW-I' ];then
python benchmark.py --lr 0.001 --n_epochs 200 --seq_length 24 --distance fft --dataset illness --batch_size 16 --ratio $ratio  --ot_type uot_mm --device cuda:4
else
python new_pipeline.py   --model $model --ratio $ratio  --dataset illness   --dataset_fold_path data/generated_datasets/italy_air_quality_rate01_step12_point   --saving_path results_point_rate01   --device cuda:5 
fi
done
done