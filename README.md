
# Generically named project ...

```
cargo run --release  1899.11s user 18.79s system 694% cpu 4:36.29 total
cargo run --release  1227.95s user 14.00s system 658% cpu 3:08.74 total # Adding pre-filtering on mz.
cargo run --release  407.91s user 14.94s system 586% cpu 1:12.08 total # Change bounding box certificate.
cargo run --release  383.80s user 14.32s system 622% cpu 1:03.99 total # Implementing count search.
cargo run --release  389.74s user 13.00s system 662% cpu 1:00.82 total # Implemented plotting and moved filter to single thread.

# cargo build --release && time ./target/release/timsextractor
# After moving to dbscan denoising
./target/release/timsextractor  479.13s user 11.96s system 725% cpu 1:07.67 total # MS2 only
./target/release/timsextractor  2681.79s user 28.76s system 724% cpu 6:14.00 total


# Only ms2 + splitting
cargo build --release && /usr/bin/time -lh ./target/release/timsextractor 
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
          2865381376  maximum resident set size
              606,702  page reclaims
                   4  page faults
               14272  voluntary context switches
              718236  involuntary context switches
       3908013776485  instructions retired
       1528190209363  cycles elapsed
          3,972,768,640  peak memory footprint
```
