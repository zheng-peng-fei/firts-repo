learning_rate=0.001
epochs=50
batch_size=128
ngpu=1
prefetch=4
script=train.py

load_pretrained='snapshots/pretrained'

test_out='tinyimages_300k'


for seed in 0; do
    for score in 'openauc'; do
      for dataset in 'cifar100' 'cifar10'; do

        results_dir="results_ood_${seed}"
        checkpoints_dir="checkpoints_ood_${seed}"

        gpu=1
        pi=1
        echo "running $score with $dataset, $test_out,  pi=$pi on GPU $gpu"
        CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$gpu CUDA_LAUNCH_BLOCKING=1 python $script $dataset \
        --model wrn --score $score --seed $seed --learning_rate $learning_rate --epochs $epochs \
        --test_out_dataset $test_out --aux_out_dataset $test_out --batch_size=$batch_size --ngpu=$ngpu \
        --prefetch=$prefetch --pi=$pi  --results_dir=$results_dir \
        --checkpoints_dir=$checkpoints_dir --load_pretrained=$load_pretrained &

        gpu=2
        pi=0.05
        echo "running $score with $dataset, $test_out,  pi=$pi on GPU $gpu"
        CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$gpu CUDA_LAUNCH_BLOCKING=1 python $script $dataset \
        --model wrn --score $score --seed $seed --learning_rate $learning_rate --epochs $epochs \
        --test_out_dataset $test_out --aux_out_dataset $test_out --batch_size=$batch_size --ngpu=$ngpu \
        --prefetch=$prefetch --pi=$pi  --results_dir=$results_dir \
        --checkpoints_dir=$checkpoints_dir --load_pretrained=$load_pretrained &

        gpu=3
        pi=0.5
        echo "running $score with $dataset, $test_out,  pi=$pi on GPU $gpu"
        CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$gpu CUDA_LAUNCH_BLOCKING=1 python $script $dataset \
        --model wrn --score $score --seed $seed --learning_rate $learning_rate --epochs $epochs \
        --test_out_dataset $test_out --aux_out_dataset $test_out --batch_size=$batch_size --ngpu=$ngpu \
        --prefetch=$prefetch --pi=$pi  --results_dir=$results_dir \
        --checkpoints_dir=$checkpoints_dir --load_pretrained=$load_pretrained &

        gpu=4
        pi=0.1
        echo "running $score with $dataset, $test_out,  pi=$pi on GPU $gpu"
        CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$gpu CUDA_LAUNCH_BLOCKING=1 python $script $dataset \
        --model wrn --score $score --seed $seed --learning_rate $learning_rate --epochs $epochs \
        --test_out_dataset $test_out --aux_out_dataset $test_out --batch_size=$batch_size --ngpu=$ngpu \
        --prefetch=$prefetch --pi=$pi  --results_dir=$results_dir \
        --checkpoints_dir=$checkpoints_dir --load_pretrained=$load_pretrained &

        gpu=5
        pi=0.2
        echo "running $score with $dataset, $test_out,  pi=$pi on GPU $gpu"
        CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$gpu CUDA_LAUNCH_BLOCKING=1 python $script $dataset \
        --model wrn --score $score --seed $seed --learning_rate $learning_rate --epochs $epochs \
        --test_out_dataset $test_out --aux_out_dataset $test_out --batch_size=$batch_size --ngpu=$ngpu \
        --prefetch=$prefetch --pi=$pi  --results_dir=$results_dir \
        --checkpoints_dir=$checkpoints_dir --load_pretrained=$load_pretrained &
        wait
      done
    done
  done
done





echo "||||||||done with training above "$1"|||||||||||||||||||"