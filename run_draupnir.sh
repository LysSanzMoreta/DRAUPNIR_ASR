#nohup ./run_draupnir.sh > ~/output.log 2>&1 &
#micromamba activate draupnir_xlstm2
nepochs=5000
nsamples=200
zdim=30

echo "UPDATED"

#CUDA_VISIBLE_DEVICES=0 python Draupnir_example.py -name simulations_src_sh3_3 -n $nepochs -n-samples $nsamples -use-blosum False -draupnir-version 1 -bsize 1 -use-cuda True -generate-samples False -prediction-method test_batched_train_full -z-dim $zdim
#CUDA_VISIBLE_DEVICES=0 python Draupnir_example.py -name simulations_src_sh3_3 -n $nepochs -n-samples $nsamples -use-blosum False -draupnir-version 1 -bsize 50 -use-cuda True -generate-samples False -prediction-method test_batched_train_full -z-dim $zdim
#CUDA_VISIBLE_DEVICES=0 python Draupnir_example.py -name simulations_src_sh3_3 -n $nepochs -n-samples $nsamples -use-blosum False -draupnir-version 1nbA -bsize 1 -use-cuda True -generate-samples False -prediction-method test_batched_train_full -z-dim $zdim
#


CUDA_VISIBLE_DEVICES=0 python Draupnir_example.py -name simulations_src_sh3_3 -n $nepochs -n-samples $nsamples -use-blosum False -draupnir-version 1nbA -prior-experiment 5 -bsize 1 -use-cuda True -generate-samples False -prediction-method test_batched_train_full -z-dim $zdim
CUDA_VISIBLE_DEVICES=0 python Draupnir_example.py -name simulations_src_sh3_3 -n $nepochs -n-samples $nsamples -use-blosum False -draupnir-version 1bB -prior-experiment 5 -bsize 50 -use-cuda True -generate-samples False -prediction-method test_batched_train_full -z-dim $zdim
CUDA_VISIBLE_DEVICES=0 python Draupnir_example.py -name simulations_src_sh3_3 -n $nepochs -n-samples $nsamples -use-blosum True -draupnir-version 1nbA -prior-experiment 5 -bsize 1 -use-cuda True -generate-samples False -prediction-method test_batched_train_full -z-dim $zdim
CUDA_VISIBLE_DEVICES=0 python Draupnir_example.py -name simulations_src_sh3_3 -n $nepochs -n-samples $nsamples -use-blosum True -draupnir-version 1bB -prior-experiment 5 -bsize 50 -use-cuda True -generate-samples False -prediction-method test_batched_train_full -z-dim $zdim



CUDA_VISIBLE_DEVICES=0 python Draupnir_example.py -name simulations_PIGBOS_1 -n $nepochs -n-samples $nsamples -use-blosum False -draupnir-version 1nbA -prior-experiment 5 -bsize 1 -use-cuda True -generate-samples False -prediction-method test_batched_train_full -z-dim $zdim
CUDA_VISIBLE_DEVICES=0 python Draupnir_example.py -name simulations_PIGBOS_1 -n $nepochs -n-samples $nsamples -use-blosum False -draupnir-version 1bB -prior-experiment 5 -bsize 50 -use-cuda True -generate-samples False -prediction-method test_batched_train_full -z-dim $zdim
CUDA_VISIBLE_DEVICES=0 python Draupnir_example.py -name simulations_PIGBOS_1 -n $nepochs -n-samples $nsamples -use-blosum True -draupnir-version 1nbA -prior-experiment 5 -bsize 1 -use-cuda True -generate-samples False -prediction-method test_batched_train_full -z-dim $zdim
CUDA_VISIBLE_DEVICES=0 python Draupnir_example.py -name simulations_PIGBOS_1 -n $nepochs -n-samples $nsamples -use-blosum True -draupnir-version 1bB -prior-experiment 5 -bsize 50 -use-cuda True -generate-samples False -prediction-method test_batched_train_full -z-dim $zdim


CUDA_VISIBLE_DEVICES=0 python Draupnir_example.py -name simulations_1GMM -n $nepochs -n-samples $nsamples -use-blosum False -draupnir-version 1nbA -prior-experiment 5 -bsize 1 -use-cuda True -generate-samples False -prediction-method test_batched_train_full -z-dim $zdim
CUDA_VISIBLE_DEVICES=0 python Draupnir_example.py -name simulations_1GMM -n $nepochs -n-samples $nsamples -use-blosum False -draupnir-version 1bB -prior-experiment 5 -bsize 50 -use-cuda True -generate-samples False -prediction-method test_batched_train_full -z-dim $zdim
CUDA_VISIBLE_DEVICES=0 python Draupnir_example.py -name simulations_1GMM -n $nepochs -n-samples $nsamples -use-blosum True -draupnir-version 1nbA -prior-experiment 5 -bsize 1 -use-cuda True -generate-samples False -prediction-method test_batched_train_full -z-dim $zdim
CUDA_VISIBLE_DEVICES=0 python Draupnir_example.py -name simulations_1GMM -n $nepochs -n-samples $nsamples -use-blosum True -draupnir-version 1bB -prior-experiment 5 -bsize 50 -use-cuda True -generate-samples False -prediction-method test_batched_train_full -z-dim $zdim