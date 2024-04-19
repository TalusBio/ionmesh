
# Peakachu

## Usage


```
cargo build --release  --features par_dataprep
./target/release/peakachu --help

RUST_LOG=info ./target/release/peakachu ...
```

## Ideas
 
- Add offset 
- keep unagregated peaks
- add 1% filter

## Performance

```
cargo run --release  1899.11s user 18.79s system 694% cpu 4:36.29 total
cargo run --release  1227.95s user 14.00s system 658% cpu 3:08.74 total # Adding pre-filtering on mz.
cargo run --release  407.91s user 14.94s system 586% cpu 1:12.08 total # Change bounding box certificate.
cargo run --release  383.80s user 14.32s system 622% cpu 1:03.99 total # Implementing count search.
cargo run --release  389.74s user 13.00s system 662% cpu 1:00.82 total # Implemented plotting and moved filter to single thread.

# cargo build --release && time ./target/release/peakachu
# After moving to dbscan denoising
./target/release/peakachu  479.13s user 11.96s system 725% cpu 1:07.67 total # MS2 only
./target/release/peakachu  2681.79s user 28.76s system 724% cpu 6:14.00 total


# Only ms2 + splitting
cargo build --release && /usr/bin/time -lh ./target/release/peakachu 
        1m18.01s real           8m4.77s user            11.41s sys
          2949349376  maximum resident set size
              694,628  page reclaims
                   5  page faults
               16024  voluntary context switches
              668389  involuntary context switches
       2435814934281  instructions retired
       1387001725171  cycles elapsed
          4859898368  peak memory footprint

# First splitting the frames...
possible oprimization: split frames without making a dense rep of the peaks. (implement frame section with scan offset)
... maybe later ...
        1m41.24s real           10m29.08s user          14.52s sys
          5595365376  maximum resident set size
           2,395,108  page reclaims
                   4  page faults
               16377  voluntary context switches
              907012  involuntary context switches
       4433,147,446,666  instructions retired
       1752673168780  cycles elapsed
          7639286144  peak memory footprint


+ Some cleanup in memory usage
        1m28.77s real           8m49.13s user           15.34s sys
          4269408256  maximum resident set size
             2,609,256  page reclaims
                   4  page faults
               16364  voluntary context switches
              841,316  involuntary context switches
       3985,072,152,374  instructions retired
       1487550309281  cycles elapsed
          7,997,342,464  peak memory footprint

# Major mem cleanup using lazy splitting of frames
        1m27.98s real           9m52.28s user           8.98s sys
          2,865,381,376  maximum resident set size
              606,702  page reclaims
                   4  page faults
               14,272  voluntary context switches
              718,236  involuntary context switches
       3,908,013,776,485  instructions retired
       1,528,190,209,363  cycles elapsed
          3,972,768,640  peak memory footprint

# Refactoring and change in tree parameters
        1m7.75s real            7m9.05s user            4.14s sys
          2,074,181,632  maximum resident set size
              596,675  page reclaims
                   6  page faults
               15,918  voluntary context switches
              586,899  involuntary context switches
       4,402,843,850,816  instructions retired
       1,162,150,354,869  cycles elapsed
          3,997,115,328  peak memory footprint


# Added tracing in time
 INFO  peakachu::utils      > Time elapsed in 'Denoising all MS2 frames' is: 57s
 INFO  peakachu::utils      > Time elapsed in 'Tracing peaks in time' is: 115s
        2m54.76s real           8m36.68s user           5.18s sys
          2444378112  maximum resident set size
             1,038,443  page reclaims
                   9  page faults
               16006  voluntary context switches
              503,127  involuntary context switches
       5379661663772  instructions retired
       1532030532305  cycles elapsed
          3,958,694,720  peak memory footprint


# Added Paralel processing of tracing --features par_dataprep
        1m51.15s real           10m18.07s user          18.94s sys
          3764240384  maximum resident set size
             2,412,027  page reclaims
                   5  page faults
               15949  voluntary context switches
             1,510,616  involuntary context switches
       5,411,367,473,183  instructions retired
       1,754,696,245,831  cycles elapsed
          5,508,632,704  peak memory footprint

# Adding peaks initial (bad) implementation of pseudo-spectrum generation
       2m21.58s real           10m47.15s user          7.71s sys
       5,291,409,408  maximum resident set size
           1,072,507  page reclaims
                   5  page faults
               16008  voluntary context switches
              990886  involuntary context switches
   5,759,247,615,620  instructions retired
   1,855,415,105,617  cycles elapsed
       5,662,807,296  peak memory footprint
```

# Added sage ...
Number of psms at 0.01 FDR: 7700
Number of peptides at 0.01 FDR: 6633
Number of proteins at 0.01 FDR: 1662
        11m52.60s real          21m55.41s user          3m56.36s sys
          4890738688  maximum resident set size
            15480303  page reclaims
                  10  page faults
               38742  voluntary context switches
             4289816  involuntary context switches
      11995716585083  instructions retired
       4702159480351  cycles elapsed
         10057480832  peak memory footprint


