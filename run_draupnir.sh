



python Draupnir_example.py -name simulations_1GMM -n 10000 -n-samples 200 -use-blosum True -use-cuda True -draupnir-version 1 -bsize 100 -build False #we only have 4000 epochs
python Draupnir_example.py -name simulations_1GMM -n 10000 -n-samples 200 -use-blosum False -use-cuda True -draupnir-version 1 -bsize 100
python Draupnir_example.py -name simulations_1GMM -n 10000 -n-samples 200 -use-blosum False -use-cuda True -draupnir-version 3a -bsize 100
python Draupnir_example.py -name simulations_1GMM -n 10000 -n-samples 200 -use-blosum False -use-cuda True -draupnir-version 3b -bsize 100 #already done

python Draupnir_example.py -name simulations_src_sh3_3 -n 10000 -n-samples 200 -use-blosum True -draupnir-version 1 -bsize 50
python Draupnir_example.py -name simulations_src_sh3_3 -n 10000 -n-samples 200 -use-blosum False -draupnir-version 1 -bsize 50
python Draupnir_example.py -name simulations_src_sh3_3 -n 10000 -n-samples 200 -use-blosum False -draupnir-version 3a -bsize 50
python Draupnir_example.py -name simulations_src_sh3_3 -n 10000 -n-samples 200 -use-blosum False -draupnir-version 3b -bsize 50


python Draupnir_example.py -name simulations_src_sh3_2 -n 10000 -n-samples 200 -use-blosum True -draupnir-version 1 -bsize 50
python Draupnir_example.py -name simulations_src_sh3_2 -n 10000 -n-samples 200 -use-blosum False -draupnir-version 1 -bsize 50
python Draupnir_example.py -name simulations_src_sh3_2 -n 10000 -n-samples 200 -use-blosum False -draupnir-version 1 -bsize 50
python Draupnir_example.py -name simulations_src_sh3_2 -n 10000 -n-samples 200 -use-blosum False -draupnir-version 1 -bsize 50


python Draupnir_example.py -name simulations_src_sh3_1 -n 10000 -n-samples 200 -use-blosum True -draupnir-version 1 -bsize 1
python Draupnir_example.py -name simulations_src_sh3_1 -n 10000 -n-samples 200 -use-blosum False -draupnir-version 1 -bsize 1
python Draupnir_example.py -name simulations_src_sh3_1 -n 10000 -n-samples 200 -use-blosum False -draupnir-version 3a -bsize 1
python Draupnir_example.py -name simulations_src_sh3_1 -n 10000 -n-samples 200 -use-blosum False -draupnir-version 3b -bsize 1

#150 leaves
python Draupnir_example.py -name simulations_sirtuins_1 -n 10000 -n-samples 200 -use-blosum True -draupnir-version 1 -bsize 50
python Draupnir_example.py -name simulations_sirtuins_1 -n 10000 -n-samples 200 -use-blosum False -draupnir-version 1 -bsize 50
python Draupnir_example.py -name simulations_sirtuins_1 -n 10000 -n-samples 200 -use-blosum False -draupnir-version 3a -bsize 50
python Draupnir_example.py -name simulations_sirtuins_1 -n 10000 -n-samples 200 -use-blosum False -draupnir-version 3b -bsize 50

#300 leaves
python Draupnir_example.py -name simulations_PIGBOS_1 -n 10000 -n-samples 200 -use-blosum True -draupnir-version 1 -bsize 50
python Draupnir_example.py -name simulations_PIGBOS_1 -n 10000 -n-samples 200 -use-blosum False -draupnir-version 1 -bsize 50
python Draupnir_example.py -name simulations_PIGBOS_1 -n 10000 -n-samples 200 -use-blosum False -draupnir-version 3a -bsize 50
python Draupnir_example.py -name simulations_PIGBOS_1 -n 10000 -n-samples 200 -use-blosum False -draupnir-version 3b -bsize 50