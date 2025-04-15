
for ratio in 0.1 0.3 0.5 0.7
do
for model in 'PSW-I'   
do
python benchmark.py --lr 0.001 --n_epochs 200 --seq_length 24 --distance fft --dataset illness --batch_size 16 --ratio $ratio --mva_kernel 5 --ot_type uot_mm --device cuda:4

done
done