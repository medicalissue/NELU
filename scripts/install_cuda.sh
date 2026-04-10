#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
#  Install NVIDIA driver + CUDA 12.4 on fresh Ubuntu 22.04.
#
#  ONLY run this if you launched a plain Ubuntu AMI — the AWS Deep
#  Learning AMI already has drivers and this script is unnecessary.
#
#  After this finishes, REBOOT the instance, then run bootstrap_instance.sh.
# ═══════════════════════════════════════════════════════════════════

set -euo pipefail

if command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi already present:"
    nvidia-smi
    echo ""
    echo "If you still want to reinstall, remove the existing driver first:"
    echo "  sudo apt-get purge -y 'nvidia-*' && sudo reboot"
    exit 0
fi

echo "══ 1. apt prereqs ══"
sudo apt-get update
sudo apt-get install -y build-essential dkms linux-headers-$(uname -r) \
    wget curl gnupg software-properties-common

echo "══ 2. CUDA keyring ══"
cd /tmp
wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
rm cuda-keyring_1.1-1_all.deb

echo "══ 3. CUDA 12.4 + driver ══"
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-4 nvidia-driver-550

echo "══ 4. PATH setup ══"
grep -q "/usr/local/cuda/bin" ~/.bashrc || cat >> ~/.bashrc <<'EOF'

# CUDA
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
EOF

echo ""
echo "══ DONE ══"
echo "  Reboot the instance now:"
echo "      sudo reboot"
echo ""
echo "  After it comes back, verify with:"
echo "      nvidia-smi"
echo ""
echo "  Then run:"
echo "      bash scripts/bootstrap_instance.sh"
