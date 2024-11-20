for ratio in 0.1 0.3 0.5 0.7
do
for model in   'PSW-I'    
do
if [ $model = 'PSW-I' ];then
python benchmark_sinkhornfft_val.py --lr 0.001 --batch_size 128 --dataset exchange --n_epochs 200 --seq_length 24 --distance fft --ratio $ratio --device cuda:5  --ot_type uot_mm 
else
python new_pipeline.py  --model $model --ratio $ratio  --dataset exchange   --dataset_fold_path data/generated_datasets/italy_air_quality_rate01_step12_point   --saving_path results_point_rate01   --device cuda:5
fi
done
done