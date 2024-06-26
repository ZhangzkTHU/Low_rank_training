
# export CUDA_VISIBLE_DEVICES=0
# nohup python train_cifar10.py --exp-name baseline > device0.out 2>&1 &

# export CUDA_VISIBLE_DEVICES=1
# nohup python train_cifar10.py --exp-name r40 --net my_vit_factorized > device1.out 2>&1 &

# export CUDA_VISIBLE_DEVICES=3
# nohup python train_cifar10.py --exp-name op --net my_vit_overparameterized > device2.out 2>&1 &

# export CUDA_VISIBLE_DEVICES=1
# nohup python train_cifar10.py --exp-name r40_2layer --net my_vit_factorized > device1.out 2>&1 &


# export CUDA_VISIBLE_DEVICES=2
# nohup python train_cifar10.py --exp-name r40_lastFF --net my_vit_factorized > device2.out 2>&1 &


# export CUDA_VISIBLE_DEVICES=0
python train_cifar10.py --log-dir logs_new --exp-name baseline --net vit_base --preempt

python train_cifar10.py --log-dir tst --exp-name ff_layer --net vit_base_factorized --preempt --r 40 --ff-layer 0

python train_cifar10.py --log-dir tst --exp-name ff_layer3 --ff-layer 3 --net vit_base_factorized --preempt --r 40 