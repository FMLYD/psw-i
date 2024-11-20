for ratio in  0.1 0.3 0.5 0.7
do
for model in  'PSW-I'   
do
if [ $model = 'PSW-I' ];then
python benchmark.py --lr 0.01 --n_epochs 200 --seq_length 24 --distance fft --dataset weather --batch_size 256 --ratio $ratio --device cuda:6 
else
python new_pipeline.py --model $model --ratio $ratio  --dataset weather   --dataset_fold_path data/generated_datasets/italy_air_quality_rate01_step12_point   --saving_path results_point_rate01   --device cuda:0
fi
done
done