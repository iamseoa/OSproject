CC = gcc
CFLAGS = -lpthread

TARGETS = baseline st sp mt mp mpmt_mutex mpmt_noSync

all: $(TARGETS)

baseline: baseline.c
	$(CC) $(CFLAGS) -o $@ $<

st: st.c
	$(CC) $(CFLAGS) -o $@ $<

sp: sp.c
	$(CC) $(CFLAGS) -o $@ $<

mt: mt.c
	$(CC) $(CFLAGS) -o $@ $<

mp: mp.c
	$(CC) $(CFLAGS) -o $@ $<

mpmt_mutex: mpmt_mutex.c
	$(CC) $(CFLAGS) -o $@ $<

mpmt_noSync: mpmt_noSync.c
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f $(TARGETS) *.o gmon.out
