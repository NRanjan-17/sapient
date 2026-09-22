# Minimal build for the SAPIENT C example.
#
#   cargo build --release -p sapient-capi   # from the repo root, first
#   make && ./c-chat "Why is the sky blue?"
#
# Once sapient.pc is installed you can replace INCLUDES/LIBS with:
#   $(shell pkg-config --cflags --libs sapient)

ROOT      := ../..
INCLUDES  := -I$(ROOT)/crates/sapient-capi/include
LIBDIR    := $(ROOT)/target/release
CFLAGS    := -std=c11 -Wall -Wextra -O2 $(INCLUDES)

UNAME := $(shell uname -s)
ifeq ($(UNAME),Darwin)
    # Rust staticlibs on macOS need the system frameworks their deps reference.
    LDFLAGS := -L$(LIBDIR) -lsapient -framework CoreFoundation -framework SystemConfiguration \
               -framework Security -framework Metal -framework QuartzCore -lc++ -liconv
    RPATH   := -Wl,-rpath,$(LIBDIR)
else
    LDFLAGS := -L$(LIBDIR) -lsapient -lpthread -ldl -lm
    RPATH   := -Wl,-rpath,$(LIBDIR)
endif

c-chat: main.c
	$(CC) $(CFLAGS) -o $@ $< $(RPATH) $(LDFLAGS)

# Link against the static archive instead of the shared library.
static: main.c
	$(CC) $(CFLAGS) -o c-chat $< $(LIBDIR)/libsapient.a $(LDFLAGS:-lsapient=)

clean:
	rm -f c-chat

.PHONY: clean static
