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
Exact same setting, only change record steps
step 2500:
{"step": 2500, "rouge_result": {"rouge-1": 23.231, "rouge-2": 8.4483, "rouge-l": 21.2772}, "val_loss": 3.7508084715020016}
{"step": 5000, "rouge_result": {"rouge-1": 23.8927, "rouge-2": 8.7269, "rouge-l": 21.908}, "val_loss": 3.6834324110716885}

step 500:
{"step": 500, "rouge_result": {"rouge-1": 16.0274, "rouge-2": 5.6842, "rouge-l": 15.1189}, "val_loss": 4.294867858442504}
{"step": 1000, "rouge_result": {"rouge-1": 16.9282, "rouge-2": 5.9364, "rouge-l": 15.798}, "val_loss": 4.242294466998726}
{"step": 1500, "rouge_result": {"rouge-1": 17.2058, "rouge-2": 6.1253, "rouge-l": 16.132}, "val_loss": 4.232869999203967}
{"step": 2000, "rouge_result": {"rouge-1": 17.8846, "rouge-2": 6.4356, "rouge-l": 16.8057}, "val_loss": 4.166847195687773}
{"step": 2500, "rouge_result": {"rouge-1": 17.3487, "rouge-2": 5.8764, "rouge-l": 16.1814}, "val_loss": 4.278883221888647}
{"step": 3000, "rouge_result": {"rouge-1": 17.9027, "rouge-2": 6.0271, "rouge-l": 16.5728}, "val_loss": 4.199275609534206}
{"step": 3500, "rouge_result": {"rouge-1": 17.7111, "rouge-2": 5.5464, "rouge-l": 16.3114}, "val_loss": 4.229530212313501}
{"step": 4000, "rouge_result": {"rouge-1": 18.282, "rouge-2": 5.885, "rouge-l": 16.9308}, "val_loss": 4.270490399267649}

What the fuck?
The weird part is for step 2500 below, it cannot reach the 2500 result above...
Perhaps the generating process did something to it... maybe I will have to cache different models...
Or just let it run for a night and see what becomes of it.

Don't generate label, just save model, same parameter:
{
    "500": {"rouge-1": 16.033, "rouge-2": 5.6823, "rouge-l": 15.1245, "val_loss": 4.294867858442504},
    "1000": {"rouge-1": 16.9175, "rouge-2": 5.941, "rouge-l": 15.7886, "val_loss": 4.242294466998726},
    "1500": {"rouge-1": 17.2255, "rouge-2": 6.1298, "rouge-l": 16.146, "val_loss": 4.232869999203967},
    "2000": {"rouge-1": 17.8999, "rouge-2": 6.4498, "rouge-l": 16.8233, "val_loss": 4.166847195687773},
    "2500": {"rouge-1": 17.341, "rouge-2": 5.8714, "rouge-l": 16.1718, "val_loss": 4.278883221888647},
    "3000": {"rouge-1": 17.9224, "rouge-2": 6.0265, "rouge-l": 16.5886, "val_loss": 4.199275609534206},
    "3500": {"rouge-1": 17.7231, "rouge-2": 5.5515, "rouge-l": 16.3147, "val_loss": 4.229530212313501},
    "4000": {"rouge-1": 18.2848, "rouge-2": 5.8882, "rouge-l": 16.9309, "val_loss": 4.270490399267649},
    "4500": {"rouge-1": 15.1165, "rouge-2": 4.9307, "rouge-l": 14.2419, "val_loss": 4.347352406864083},
    "5000": {"rouge-1": 17.3709, "rouge-2": 5.307, "rouge-l": 16.36, "val_loss": 4.305379054084789}
}
How.
Do note that it matched the script with generation close enough, and val loss have exact match. I think generation won't touch the settings...

Maybe because...it was broken between? Maybe I HAVE to set record_step 2500.