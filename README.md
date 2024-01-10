
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
                   0  average shared memory size
                   0  average unshared data size
                   0  average unshared stack size
              694628  page reclaims
                   5  page faults
                   0  swaps
                   0  block input operations
                   0  block output operations
                   0  messages sent
                   0  messages received
                   0  signals received
               16024  voluntary context switches
              668389  involuntary context switches
       2435814934281  instructions retired
       1387001725171  cycles elapsed
          4859898368  peak memory footprint
```
