
This tensor is created by the following code for reproducibility.

```python
import torch

# struct
generator = torch.Generator().manual_seed(0)
# T+1=26, B=32, d=8*16*16*16
shape = (26, 32, 8, 16, 16, 16) 
latents_struct = torch.randn(shape, generator=generator)
torch.save(latents_struct, "eval_noise/struct_tensor.pt")

# slat
generator = torch.Generator().manual_seed(0)
# T+1=26, B=32, d=8*30000
shape = (26, 32, 8, 30000)
latents_slat = torch.randn(shape, generator=generator)
torch.save(latents_slat, "eval_noise/slat_tensor.pt")
```

