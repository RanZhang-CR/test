run:
	gcc -w -Wall -mavx2 -mfma -O2 main.c kernel.h -o knn.x -DN=22687 -DNUM_A=109410 -fopenmp -std=c99
	./knn.x

run_k1:
	gcc -w -Wall -mavx2 -mfma -O2 main_kernel1_performance.c kernel.h -o knnk2.x -DN=22687 -DNUM_A=109410 -fopenmp -std=c99
	./knnk1.x

run_k2:
	gcc -w -Wall -mavx2 -mfma -O2 main_kernel2_performance.c kernel.h -o knnk2.x -DN=22687 -DNUM_A=109410 -fopenmp -std=c99
	./knnk2.x

run_k3:
	gcc -w -Wall -mavx2 -mfma -O2 main_kernel3_performance.c kernel.h -o knnk3.x -DN=22687 -DNUM_A=109410 -fopenmp -std=c99
	./knnk3.x

cleanup:
	rm -rf *~
	rm -rf *.x

