import torch
import time

print("PyTorch version:", torch.__version__)
print("MPS available:", hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
print("MPS built:", hasattr(torch.backends, "mps") and torch.backends.mps.is_built())

# Kích thước ma trận (có thể tăng để thấy rõ hơn)
N = 5000
a = torch.rand(N, N)
b = torch.rand(N, N)
myiters= 100

def benchmark(device, a, b, iters=myiters):
    a = a.to(device)
    b = b.to(device)
    torch.mps.synchronize() if device.type == "mps" else None
    torch.cuda.synchronize() if device.type == "cuda" else None

    start = time.time()
    for _ in range(iters):
        c = a @ b
    if device.type == "mps":
        torch.mps.synchronize()
    if device.type == "cuda":
        torch.cuda.synchronize()
    end = time.time()
    return (end - start) / iters

# CPU benchmark
cpu_time = benchmark(torch.device("cpu"), a, b)
print(f"CPU avg time ({myiters} runs): {cpu_time:.4f} s")

# MPS benchmark
if torch.backends.mps.is_available():
    mps_time = benchmark(torch.device("mps"), a, b)
    print(f"MPS avg time ({myiters} runs): {mps_time:.4f} s")
    print(f"➡️  MPS faster by ~{cpu_time/mps_time:.2f}x")
