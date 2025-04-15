for ratio in 0.1 0.3 0.5 0.7
do
for model in    'PSW-I'   
do
python benchmark.py --lr 0.01 --batch_size 256 --dataset ETT_m2 --n_epochs 200 --seq_length 24 --distance fft --ratio $ratio --device cuda:3 --mva_kernel 3 --ot_type uot_mm --reg_m 1 --dropout 0

done
done