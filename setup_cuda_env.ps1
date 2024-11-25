# Update and upgrade the system
Write-Output "Updating and upgrading the system..."
Start-Process "powershell" -ArgumentList "sudo apt-get update && sudo apt-get upgrade -y" -Wait

# Install Python pip
Write-Output "Installing Python3 pip..."
Start-Process "powershell" -ArgumentList "sudo apt-get install python3-pip -y" -Wait

# Install necessary packages
Write-Output "Installing essential packages..."
Start-Process "powershell" -ArgumentList "sudo apt-get install -y ubuntu-drivers-common alsa-utils" -Wait

# Install NVIDIA driver
Write-Output "Installing NVIDIA driver..."
Start-Process "powershell" -ArgumentList "sudo apt-get install -y nvidia-driver-535" -Wait

# Auto-install additional drivers
Write-Output "Running ubuntu-drivers autoinstall..."
Start-Process "powershell" -ArgumentList "sudo ubuntu-drivers autoinstall" -Wait

# Check NVIDIA driver installation
Write-Output "Checking NVIDIA driver installation..."
Start-Process "powershell" -ArgumentList "nvidia-smi" -Wait

# Download and install CUDA toolkit
Write-Output "Downloading CUDA installer..."
Start-Process "powershell" -ArgumentList "wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda_12.4.0_550.54.14_linux.run -O cuda_installer.run" -Wait

Write-Output "Installing CUDA toolkit..."
Start-Process "powershell" -ArgumentList "sudo sh cuda_installer.run --silent --toolkit" -Wait

# Update environment variables
Write-Output "Updating environment variables for CUDA..."
$CUDA_Exports = "export PATH=/usr/local/cuda-12.4/bin`$PATH:/usr/local/cuda-12.4/lib64"
Add-Content -Path "$HOME/.bashrc" -Value $CUDA_Exports

# Source the updated bashrc
Write-Output "Applying environment variables..."
Start-Process "powershell" -ArgumentList "source ~/.bashrc" -Wait

# Verify CUDA installation
Write-Output "Verifying CUDA installation..."
Start-Process "powershell" -ArgumentList "nvcc -V" -Wait

# Install PyTorch
Write-Output "Installing PyTorch..."
Start-Process "powershell" -ArgumentList "pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124" -Wait

# Install cuDNN
Write-Output "Installing cuDNN..."
Start-Process "powershell" -ArgumentList "sudo apt install -y nvidia-cudnn" -Wait

Write-Output "Setup completed! Please restart the shell or reboot the system if necessary."
