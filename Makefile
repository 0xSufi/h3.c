CC := clang
AR := ar
CFLAGS := -std=c11 -O3 -Wall -Wextra -Wpedantic -Wshadow \
	-Wconversion -Wno-sign-conversion -D_DARWIN_C_SOURCE
OBJCFLAGS := $(CFLAGS) -fobjc-arc
FRAMEWORKS := -framework Foundation -framework Metal
LDLIBS := $(FRAMEWORKS) -lm

LIB_C := h3.c h3_host.c h3_safetensors.c
LIB_M := h3_metal.m
LIB_OBJ := $(LIB_C:.c=.o) $(LIB_M:.m=.o)

.PHONY: all test clean

all: h3 libh3.a

h3: main.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

libh3.a: $(LIB_OBJ)
	$(AR) rcs $@ $^

h3_tests: tests/test_h3.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

test: h3_tests
	./h3_tests

%.o: %.c
	$(CC) $(CFLAGS) -I. -c $< -o $@

%.o: %.m
	$(CC) $(OBJCFLAGS) -I. -c $< -o $@

tests/%.o: tests/%.c
	$(CC) $(CFLAGS) -I. -c $< -o $@

clean:
	rm -f h3 h3_tests libh3.a *.o tests/*.o
