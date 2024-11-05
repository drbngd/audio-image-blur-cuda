SOURCES = audioblur.cu audioblur.cpp

NVCC = nvcc

ifeq ($(OS_SIZE),32)
NVCC_FLAGS := -m32
else
NVCC_FLAGS := -m64
endif
NVCC_FLAGS += -g -diag-suppress 177 -Wno-deprecated-gpu-targets --expt-relaxed-constexpr 

INCLUDE_DIR := -I/storage/ice-shared/ece3803gpu/libsndfile/include
LIB_DIR := -L/storage/ice-shared/ece3803gpu/libsndfile
LIBS := -lsndfile

all: blur-audio blur-noaudio

blur-audio:  $(SOURCES)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDE_DIR) $(LIB_DIR) -DAUDIO_ON=1 -o blur-audio $^ $(LIBS) 

blur-noaudio:  $(SOURCES)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDE_DIR) $(LIB_DIR) -DAUDIO_ON=0 -o blur-noaudio $^ $(LIBS) 


test-audio: blur-audio resources/example_test.wav
	./blur-audio 32 128 resources/example_test.wav resources/example_test_out.wav

test-noaudio: blur-noaudio
	./blur-noaudio 32 128

test: test-audio

clean:
	rm -f blur-audio blur-noaudio resources/example_test_out.wav

