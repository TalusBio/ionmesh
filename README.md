
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
```
