
# Peakachu

## Usage


```
cargo build --release  --features par_dataprep
./target/release/peakachu --help

RUST_LOG=info ./target/release/peakachu ...
```

## 

## Roadmap

1. Use aggregation metrics to re-score sage search.
2. Do a two pass speudospec generation, where the first pass finds the centroids and the second pass aggregates around a radius. (this will prevent the issue where common ions, like b2's are assigned only to the most intense spectrum in a window....)
  - RN I believe 
3. Re-define rt parmeters in the config as a function of the cycle time and not raw seconds.
4. Add targeted extraction.
5. Add detection of MS1 features + notched search instead of wide window search.
6. Change pseudo-spectrum aggregation
  - I am happy with the trace aggregation (It can maybe be generalized to handle synchro or midia).


## Ideas
 
- Add offset 
- add 1% filter

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


