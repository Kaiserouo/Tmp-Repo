# ADL-HW3

Very important note:
+ I use CUDA 11.6. (I use RTX 3060 6GB, so cannot support CUDA < 11)
+ **My tensorflow cannot access GPU.** I don't know why, but that means I can merge ROUGE score calculation with training loop. **If your tensorflow can access GPU, then it will DEFINITELY out-of-memory, even for a 16GB GPU (P100)**
  + FYI, in my case (pytorch can access GPU, tensorflow can't), the maximum GPU memory usage is 5.9GB. 

## Todo:
+ adafactor (instead of Adam)
+ fp16: it looks broken...and does not seem to save any space at all???????
+ train: have not tested yet
+ generate? what can be tweaked? what is the requirement?

## First try: 
