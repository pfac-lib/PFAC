PFAC is an open source, high-performance string matching library supporting NVIDIA GPUs.

# Introduction #

**PFAC** library is based on the _Parallel Failureless-AC_ Algorithm<sup>1</sup> proposed at the IEEE GLOBECOM 2010.


# Details #
The PFAC algorithm efficiently exploits the parallelism of the **Aho-Corasick** algorithm by creating an individual thread for each byte of an input stream to identify any pattern starting at the thread’s starting position. The number of threads created by the PFAC algorithm is equal to the length of an input stream.


<sup>1</sup>Cheng-Hung Lin, Sheng-Yu Tsai, Chen-Hsiung Liu, Shih-Chieh Chang, and Jyuo-Min Shyu, "**Accelerating String Matching Using Multi-threaded Algorithm on GPU**,"  in _Proc. IEEE GLOBAL COMMUNICATIONS CONFERENCE (IEEE GLOBECOM 2010)_, Miami, Florida, USA, December 6-10, 2010.

# People #