CC = gcc
CFLAGS = -lpthread 
SRC_DIR = src
BIN_DIR = bin

TARGETS = baseline st sp mt mp mpmt_mutex mpmt_noSync

all: $(TARGETS)

$(TARGETS):
	$(CC) $(CFLAGS) $(SRC_DIR)/$@.c -o $(BIN_DIR)/$@

clean:
	rm -f $(BIN_DIR)/* gmon.out

