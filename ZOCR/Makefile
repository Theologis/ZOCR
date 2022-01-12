### Make File of character OCR UTH AoT

PROJECT   =   OCR

CXX       :=   g++

CUR_DIR =   $(shell pwd)
SRC_DIR     = ./src
BUILD_DIR   = ./build
INC_DIR     = ./inc
MODEL   =   $(CUR_DIR)/model/dpu_chars74k_0.elf


# linking libraries of OpenCV
LDFLAGS   =   $(shell pkg-config --libs opencv )

# linking libraries of DNNDK
LDFLAGS   +=  -lhineon -ln2cube -ldputils

SRCS := $(shell find $(SRC_DIR) -name *.cpp )
OBJS := $(SRCS:%=$(BUILD_DIR)/%.o)
DEPS := $(OBJS:.o=.d)

INC_DIRS := $(shell find $(SRC_DIR) -type d)
INC_FLAGS := $(addprefix -I,$(INC_DIRS))

CPPFLAGS ?= $(INC_FLAGS) -O2 -Wall -Wpointer-arith -std=c++11 -ffast-math -MMD -MP -fopenmp

ARCH    =   $(shell uname -m | sed -e s/arm.*/armv71/ -e s/aarch64.*/aarch64/)
ifeq ($(ARCH),armv71)
    CPPFLAGS += -mcpu=cortex-a9 -mfloat-abi=hard -mfpu=neon
endif
ifeq ($(ARCH),aarch64)
    CPPFLAGS += -mcpu=cortex-a53
endif

$(PROJECT): $(OBJS)
	$(CXX) $(OBJS) $(MODEL) -o $@ $(LDFLAGS)


# c++ source
$(BUILD_DIR)/%.cpp.o: %.cpp
	$(MKDIR_P) $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@


.PHONY: clean

clean:
	$(RM) -r $(BUILD_DIR)

-include $(DEPS)

MKDIR_P ?= mkdir -p

