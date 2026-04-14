#nohup ./run_draupnir.sh > ~/output.log 2>&1 &
#micromamba activate draupnir_xlstm2
nepochs=5000
nsamples=200
zdim=30



#CUDA_VISIBLE_DEVICES=0 python Draupnir_example.py -name simulations_src_sh3_3 -n $nepochs -n-samples $nsamples -use-blosum False -draupnir-version 1 -covariance-prior og -bsize 1 -use-cuda True -generate-samples False -prediction-method test_batched_train_full -z-dim $zdim
#CUDA_VISIBLE_DEVICES=0 python Draupnir_example.py -name simulations_src_sh3_3 -n $nepochs -n-samples $nsamples -use-blosum False -draupnir-version 1 -covariance-prior og -bsize 50 -use-cuda True -generate-samples False -prediction-method test_batched_train_full -z-dim $zdim
#CUDA_VISIBLE_DEVICES=0 python Draupnir_example.py -name simulations_src_sh3_3 -n $nepochs -n-samples $nsamples -use-blosum True -draupnir-version 1 -covariance-prior og -bsize 1 -use-cuda True -generate-samples False -prediction-method test_batched_train_full -z-dim $zdim
#CUDA_VISIBLE_DEVICES=0 python Draupnir_example.py -name simulations_src_sh3_3 -n $nepochs -n-samples $nsamples -use-blosum True -draupnir-version 1 -covariance-prior og -bsize 50 -use-cuda True -generate-samples False -prediction-method test_batched_train_full -z-dim $zdim
#
#
#
#CUDA_VISIBLE_DEVICES=0 python Draupnir_example.py -name simulations_src_sh3_3 -n $nepochs -n-samples $nsamples -use-blosum False -draupnir-version 1 -covariance-prior 0 -bsize 1 -use-cuda True -generate-samples False -prediction-method test_batched_train_full -z-dim $zdim
#CUDA_VISIBLE_DEVICES=0 python Draupnir_example.py -name simulations_src_sh3_3 -n $nepochs -n-samples $nsamples -use-blosum False -draupnir-version 1 -covariance-prior 0 -bsize 50 -use-cuda True -generate-samples False -prediction-method test_batched_train_full -z-dim $zdim
#CUDA_VISIBLE_DEVICES=0 python Draupnir_example.py -name simulations_src_sh3_3 -n $nepochs -n-samples $nsamples -use-blosum True -draupnir-version 1 -covariance-prior 0 -bsize 1 -use-cuda True -generate-samples False -prediction-method test_batched_train_full -z-dim $zdim
#CUDA_VISIBLE_DEVICES=0 python Draupnir_example.py -name simulations_src_sh3_3 -n $nepochs -n-samples $nsamples -use-blosum True -draupnir-version 1 -covariance-prior 0 -bsize 50 -use-cuda True -generate-samples False -prediction-method test_batched_train_full -z-dim $zdim
#
#
#CUDA_VISIBLE_DEVICES=0 python Draupnir_example.py -name simulations_src_sh3_3 -n $nepochs -n-samples $nsamples -use-blosum False -draupnir-version 1 -covariance-prior 5 -bsize 1 -use-cuda True -generate-samples False -prediction-method test_batched_train_full -z-dim $zdim
#CUDA_VISIBLE_DEVICES=0 python Draupnir_example.py -name simulations_src_sh3_3 -n $nepochs -n-samples $nsamples -use-blosum False -draupnir-version 1 -covariance-prior 5 -bsize 50 -use-cuda True -generate-samples False -prediction-method test_batched_train_full -z-dim $zdim
#CUDA_VISIBLE_DEVICES=0 python Draupnir_example.py -name simulations_src_sh3_3 -n $nepochs -n-samples $nsamples -use-blosum True -draupnir-version 1 -covariance-prior 5 -bsize 1 -use-cuda True -generate-samples False -prediction-method test_batched_train_full -z-dim $zdim
#CUDA_VISIBLE_DEVICES=0 python Draupnir_example.py -name simulations_src_sh3_3 -n $nepochs -n-samples $nsamples -use-blosum True -draupnir-version 1 -covariance-prior 5 -bsize 50 -use-cuda True -generate-samples False -prediction-method test_batched_train_full -z-dim $zdim



CUDA_VISIBLE_DEVICES=0 python Draupnir_example.py -name simulations_1GMM -n $nepochs -n-samples $nsamples -use-blosum False -draupnir-version 1 -covariance-prior 5 -bsize 50 -use-cuda True -generate-samples False -prediction-method test_batched_train_full -z-dim $zdim
CUDA_VISIBLE_DEVICES=0 python Draupnir_example.py -name simulations_1GMM -n $nepochs -n-samples $nsamples -use-blosum True -draupnir-version 1 -covariance-prior 5 -bsize 50 -use-cuda True -generate-samples False -prediction-method test_batched_train_full -z-dim $zdim


CUDA_VISIBLE_DEVICES=0 python Draupnir_example.py -name simulations_1GMM -n $nepochs -n-samples $nsamples -use-blosum False -draupnir-version 1 -covariance-prior og -bsize 50 -use-cuda True -generate-samples False -prediction-method test_batched_train_full -z-dim $zdim
CUDA_VISIBLE_DEVICES=0 python Draupnir_example.py -name simulations_1GMM -n $nepochs -n-samples $nsamples -use-blosum True -draupnir-version 1 -covariance-prior og -bsize 50 -use-cuda True -generate-samples False -prediction-method test_batched_train_full -z-dim $zdim

CUDA_VISIBLE_DEVICES=0 python Draupnir_example.py -name simulations_1GMM -n $nepochs -n-samples $nsamples -use-blosum False -draupnir-version 1 -covariance-prior 0 -bsize 50 -use-cuda True -generate-samples False -prediction-method test_batched_train_full -z-dim $zdim
CUDA_VISIBLE_DEVICES=0 python Draupnir_example.py -name simulations_1GMM -n $nepochs -n-samples $nsamples -use-blosum True -draupnir-version 1 -covariance-prior 0 -bsize 50 -use-cuda True -generate-samples False -prediction-method test_batched_train_full -z-dim $zdim