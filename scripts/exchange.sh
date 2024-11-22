for ratio in 0.1 0.3 0.5 0.7
do
for model in   'PSW-I'    
do
python benchmark.py --lr 0.001 --batch_size 128 --dataset exchange --n_epochs 200 --seq_length 24 --distance fft --ratio $ratio --device cuda:5  --ot_type uot_mm 

done
done