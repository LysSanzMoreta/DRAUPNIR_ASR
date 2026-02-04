

nepochs=15000
nsamples=200

python Draupnir_example.py -name simulations_1GMM -n $nepochs -n-samples $nsamples -use-blosum True -draupnir-version 1 -bsize 100 -use-cuda True -generate-samples False -prediction-method test_batched_train_full
python Draupnir_example.py -name simulations_1GMM -n $nepochs -n-samples $nsamples -use-blosum False -draupnir-version 1 -bsize 10 -use-cuda True -generate-samples False -prediction-method test_batched_train_full
python Draupnir_example.py -name simulations_1GMM -n $nepochs -n-samples $nsamples -use-blosum False -draupnir-version 3a -bsize 100 -use-cuda True -generate-samples False -prediction-method test_batched_train_full
python Draupnir_example.py -name simulations_1GMM -n $nepochs -n-samples $nsamples -use-blosum False -draupnir-version 3b -bsize 100 -use-cuda True -generate-samples False -prediction-method test_batched_train_full

python Draupnir_example.py -name simulations_src_sh3_3 -n $nepochs -n-samples $nsamples -use-blosum True -draupnir-version 1 -bsize 50 -use-cuda True -generate-samples False -prediction-method test_batched_train_full
python Draupnir_example.py -name simulations_src_sh3_3 -n $nepochs -n-samples $nsamples -use-blosum False -draupnir-version 1 -bsize 50 -use-cuda True -generate-samples False -prediction-method test_batched_train_full
python Draupnir_example.py -name simulations_src_sh3_3 -n $nepochs -n-samples $nsamples -use-blosum True -draupnir-version 1 -bsize 1 -use-cuda True -generate-samples False -prediction-method test_batched_train_full
python Draupnir_example.py -name simulations_src_sh3_3 -n $nepochs -n-samples $nsamples -use-blosum False -draupnir-version 1 -bsize 1 -use-cuda True -generate-samples False -prediction-method test_batched_train_full
python Draupnir_example.py -name simulations_src_sh3_3 -n $nepochs -n-samples $nsamples -use-blosum False -draupnir-version 3a -bsize 50 -use-cuda True -generate-samples False -prediction-method test_batched_train_full
python Draupnir_example.py -name simulations_src_sh3_3 -n $nepochs -n-samples $nsamples -use-blosum False -draupnir-version 3b -bsize 50 -use-cuda True -generate-samples False -prediction-method test_batched_train_full

python Draupnir_example.py -name simulations_src_sh3_2 -n $nepochs -n-samples $nsamples -use-blosum True -draupnir-version 1 -bsize 100 -use-cuda True -generate-samples False -prediction-method test_batched_train_full
python Draupnir_example.py -name simulations_src_sh3_2 -n $nepochs -n-samples $nsamples -use-blosum False -draupnir-version 1 -bsize 100 -use-cuda True -generate-samples False -prediction-method test_batched_train_full
python Draupnir_example.py -name simulations_src_sh3_2 -n $nepochs -n-samples $nsamples -use-blosum False -draupnir-version 1 -bsize 100 -use-cuda True -generate-samples False -prediction-method test_batched_train_full
python Draupnir_example.py -name simulations_src_sh3_2 -n $nepochs -n-samples $nsamples -use-blosum False -draupnir-version 1 -bsize 100 -use-cuda True -generate-samples False -prediction-method test_batched_train_full

python Draupnir_example.py -name simulations_src_sh3_1 -n $nepochs -n-samples $nsamples -use-blosum True -draupnir-version 1 -bsize 1 -use-cuda True -generate-samples False -prediction-method test_batched_train_full
python Draupnir_example.py -name simulations_src_sh3_1 -n $nepochs -n-samples $nsamples -use-blosum False -draupnir-version 1 -bsize 1 -use-cuda True -generate-samples False -prediction-method test_batched_train_full
python Draupnir_example.py -name simulations_src_sh3_1 -n $nepochs -n-samples $nsamples -use-blosum False -draupnir-version 3a -bsize 1 -use-cuda True -generate-samples False -prediction-method test_batched_train_full
python Draupnir_example.py -name simulations_src_sh3_1 -n $nepochs -n-samples $nsamples -use-blosum False -draupnir-version 3b -bsize 1 -use-cuda True -generate-samples False -prediction-method test_batched_train_full

#150 leaves
python Draupnir_example.py -name simulations_sirtuins_1 -n $nepochs -n-samples $nsamples -use-blosum True -draupnir-version 1 -bsize 1 -use-cuda True -generate-samples False -prediction-method test_batched_train_full
python Draupnir_example.py -name simulations_sirtuins_1 -n $nepochs -n-samples $nsamples -use-blosum False -draupnir-version 1 -bsize 1 -use-cuda True -generate-samples False -prediction-method test_batched_train_full
python Draupnir_example.py -name simulations_sirtuins_1 -n $nepochs -n-samples $nsamples -use-blosum False -draupnir-version 3a -bsize 1 -use-cuda True -generate-samples False -prediction-method test_batched_train_full
python Draupnir_example.py -name simulations_sirtuins_1 -n $nepochs -n-samples $nsamples -use-blosum False -draupnir-version 3b -bsize 1 -use-cuda True -generate-samples False -prediction-method test_batched_train_full

#300 leaves #todo: pigbos only works with draupnir version 1 and batch size 1
#python Draupnir_example.py -name simulations_PIGBOS_1 -n $nepochs -n-samples $nsamples -use-blosum True -draupnir-version 1 -bsize 1 -use-cuda True -generate-samples False -prediction-method test_batched_train_full
#python Draupnir_example.py -name simulations_PIGBOS_1 -n $nepochs -n-samples $nsamples -use-blosum False -draupnir-version 1 -bsize 1 -use-cuda True -generate-samples False -prediction-method test_batched_train_full
#python Draupnir_example.py -name simulations_PIGBOS_1 -n $nepochs -n-samples $nsamples -use-blosum True -draupnir-version 1 -bsize 50 -use-cuda True -generate-samples False -prediction-method test_batched_train_full
#python Draupnir_example.py -name simulations_PIGBOS_1 -n $nepochs -n-samples $nsamples -use-blosum False -draupnir-version 1 -bsize 50 -use-cuda True -generate-samples False -prediction-method test_batched_train_full
#python Draupnir_example.py -name simulations_PIGBOS_1 -n $nepochs -n-samples $nsamples -use-blosum False -draupnir-version 3a -bsize 50 -use-cuda True -generate-samples False -prediction-method test_batched_train_full
#python Draupnir_example.py -name simulations_PIGBOS_1 -n $nepochs -n-samples $nsamples -use-blosum False -draupnir-version 3b -bsize 50 -use-cuda True -generate-samples False -prediction-method test_batched_train_full

python Draupnir_example.py -name simulations_blactamase_1 -n $nepochs -n-samples $nsamples -use-blosum True -draupnir-version 1 -bsize 1 -use-cuda True -generate-samples False -prediction-method test_batched_train_full
python Draupnir_example.py -name simulations_blactamase_1 -n $nepochs -n-samples $nsamples -use-blosum False -draupnir-version 1 -bsize 1 -use-cuda True -generate-samples False -prediction-method test_batched_train_full
python Draupnir_example.py -name simulations_blactamase_1 -n $nepochs -n-samples $nsamples -use-blosum False -draupnir-version 3a -bsize 1 -use-cuda True -generate-samples Fals -prediction-method test_batched_train_full
python Draupnir_example.py -name simulations_blactamase_1 -n $nepochs -n-samples $nsamples -use-blosum False -draupnir-version 3b -bsize 1 -use-cuda True -generate-samples Fals -prediction-method test_batched_train_full

#50 leaves
python Draupnir_example.py -name simulations_calcitonin_1 -n $nepochs -n-samples $nsamples -use-blosum True -draupnir-version 1 -bsize 1 -use-cuda True -generate-samples False -prediction-method test_batched_train_full
python Draupnir_example.py -name simulations_calcitonin_1 -n $nepochs -n-samples $nsamples -use-blosum False -draupnir-version 1 -bsize 1 -use-cuda True -generate-samples False -prediction-method test_batched_train_full
python Draupnir_example.py -name simulations_calcitonin_1 -n $nepochs -n-samples $nsamples -use-blosum False -draupnir-version 3a -bsize 1 -use-cuda True -generate-samples False -prediction-method test_batched_train_full
python Draupnir_example.py -name simulations_calcitonin_1 -n $nepochs -n-samples $nsamples -use-blosum False -draupnir-version 3b -bsize 1 -use-cuda True -generate-samples False -prediction-method test_batched_train_full