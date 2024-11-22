for ratio in  0.1 0.3 0.5 0.7
do
for model in 'PSW-I'   
do
python benchmark.py --lr 0.01 --n_epochs 200 --ot_type uot_mm --dropout 0 --seq_length 24  --distance fft --dataset PEMS03 --batch_size 64 --ratio $ratio --device cuda:3

done
done