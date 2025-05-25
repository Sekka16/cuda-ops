all:
	# nvcc -arch=sm_80 --ptxas-options=-v sgemm.cu -o sgemm
	# nvcc -arch=sm_80 sgemm.cu -o sgemm
	# nvcc --ptxas-options=-v -arch=sm_86 sgemm.cu -o sgemm
	nvcc -Xptxas="-v" -arch=sm_86 sgemm.cu -o sgemm
	# nvcc -arch=sm_86 sgemm.cu -o sgemm
clean:
	rm -f sgemm