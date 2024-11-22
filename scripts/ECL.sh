for ratio in  0.1 0.3 0.5 0.7
do
for model in 'PSW-I'  
do
python benchmark.py --lr 0.05 --n_epochs 200 --seq_length 24 --mva_kernel 24 --distance fft --dataset Electricity --batch_size 256 --ratio $ratio --device cuda:0

done
done