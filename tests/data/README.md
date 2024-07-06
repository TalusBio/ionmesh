
# Build testing data

The main purpose of this is to have a reproducible way to generate
data for testing. This should keep the repo size small and allow for
ease of extension of the tests as we might see fit.

One of the things I am attempting to do is to keep all the data represented
in plain text and then built onto the binary formats that actually get
used. The main rationale for is is transparency, improve the utility of
the source control and mild paranoia after the XZ exploit.
