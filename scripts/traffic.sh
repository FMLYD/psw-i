for ratio in 0.1 0.3 0.5 0.7
do
for model in   'PSW-I'  
do
python benchmark.py --lr 0.2 --n_epochs 300 --ot_type uot_mm --dropout 0 --seq_length 24  --distance fft --dataset traffic --batch_size 256 --ratio $ratio --device cuda:5

done
done