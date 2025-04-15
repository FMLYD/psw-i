for ratio in 0.1  0.3 0.5 0.7
do
for model in  'PSW-I'   
do
python benchmark.py --lr 0.01 --batch_size 256 --dataset ETTh1 --n_epochs 200 --seq_length 24 --distance fft --ratio $ratio --device cuda:0 --mva_kernel 7 --ot_type uot_mm --reg_m 1

done
done