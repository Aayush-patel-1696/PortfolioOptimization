# %%
import torch.nn as nn
import torch

# %%
          
# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self, k, analysis_window=60, conv_layers=5, kernel_length=5, stride=2):
        super(Discriminator, self).__init__()
        
        layers = []
        in_channels = k
        for i in range(conv_layers):
            out_channels = k * (2 ** i)
            layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=kernel_length, stride=stride,padding=kernel_length // 2))
            layers.append(nn.LeakyReLU(0.2))
            in_channels = out_channels
        
        self.conv = nn.Sequential(*layers)
        
        # The output size depends on the input size and the convolution parameters
        conv_out_size = self._calculate_conv_output_size(analysis_window, kernel_length, stride, conv_layers)
        
        # Fully connected layer for the final output
        self.fc = nn.Sequential(
            nn.Linear(out_channels * conv_out_size, 1),
            nn.Sigmoid())

    
    def _calculate_conv_output_size(self, input_size, kernel_size, stride, num_layers):
        size = input_size
        for _ in range(num_layers):
            size = (size + 2 * ((kernel_size - 1) // 2) - (kernel_size - 1) - 1) // stride + 1
        return size
    
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # Flatten the output
        x = self.fc(x)
        return x

# Example usage
k = 5  # Example value for k
analysis_window = 60
model = Discriminator(k, analysis_window)

# Print the model
print(model)

# Example input
x = torch.randn(106, k, analysis_window)  # Batch size of 16, k input channels, and analysis window of 60
output = model(x)
print(output.shape) 

# %%



