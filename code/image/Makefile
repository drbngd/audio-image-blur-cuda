
CFLAGS= -O3 -Wextra 
NVCC=nvcc
NVCC_FLAGS= 
DEPS=


all: imageblur

imageblur: imageblur.cu util.h ppm.cu ppm.h
	$(NVCC) -o imageblur imageblur.cu ppm.cu $(NVCC_FLAGS) $(addprefix -Xcompiler ,$(CFLAGS))

clean:
	rm -rf imageblur output-mem.ppm output-tex1d.ppm output-tex2d.ppm

test: imageblur
	./imageblur 0 output-mem.ppm
	./imageblur 1 output-tex1d.ppm
	./imageblur 2 output-tex2d.ppm