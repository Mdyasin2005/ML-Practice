import torch

print("PyTorch:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())
print("CUDA Version:", torch.version.cuda)

if torch.cuda.is_available():

    print("GPU:", torch.cuda.get_device_name(0))

    x = torch.rand(3000, 3000).cuda()
    y = torch.rand(3000, 3000).cuda()

    z = torch.matmul(x, y)

    print("Device:", z.device)
    print("GPU calculation successful!")