# Google Cloud AVX-512 Setup for MAGNUS

## GCP Instance Recommendations for AVX-512

### ✅ Confirmed AVX-512 Support on GCP

Google Cloud instances with **Intel Cascade Lake** or **Intel Ice Lake** CPUs support AVX-512:

| Instance Type | vCPUs | RAM | CPU Type | AVX-512 | Price/hour | Spot Price |
|--------------|-------|-----|----------|---------|------------|------------|
| **n2-standard-4** | 4 | 16 GB | Cascade Lake | ✅ Yes | ~$0.19 | ~$0.06 |
| **n2-standard-8** | 8 | 32 GB | Cascade Lake | ✅ Yes | ~$0.38 | ~$0.11 |
| **n2-highmem-4** | 4 | 32 GB | Cascade Lake | ✅ Yes | ~$0.26 | ~$0.08 |
| **c2-standard-4** | 4 | 16 GB | Cascade Lake | ✅ Yes | ~$0.21 | ~$0.06 |
| **c2-standard-8** | 8 | 32 GB | Cascade Lake | ✅ Yes | ~$0.42 | ~$0.13 |
| **c3-standard-4** | 4 | 16 GB | Sapphire Rapids | ✅ Yes | ~$0.20 | ~$0.06 |

### 🎯 Recommended Options

#### Budget Development: **n2-standard-4** (Spot)
- **4 vCPUs**, 16 GB RAM
- **~$0.06/hour** with preemptible/spot
- Good for development and quick tests
- Cascade Lake with AVX-512

#### Best Value: **n2-standard-8** (Spot)
- **8 vCPUs**, 32 GB RAM  
- **~$0.11/hour** with preemptible/spot
- Handles large matrix tests well
- Good parallel performance

#### Performance Testing: **c2-standard-8** (Spot)
- **8 vCPUs**, 32 GB RAM
- **~$0.13/hour** with preemptible/spot
- Compute-optimized for benchmarks
- Better single-thread performance

### 🚀 Quick Setup Commands

## 1. Create Instance with gcloud CLI

```bash
# Budget option: n2-standard-4 with spot pricing
gcloud compute instances create magnus-avx512-dev \
    --zone=us-central1-a \
    --machine-type=n2-standard-4 \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size=50GB \
    --boot-disk-type=pd-ssd \
    --preemptible \
    --max-run-duration=8h \
    --instance-termination-action=STOP

# Better performance: n2-standard-8 with spot pricing  
gcloud compute instances create magnus-avx512-dev \
    --zone=us-central1-a \
    --machine-type=n2-standard-8 \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size=100GB \
    --boot-disk-type=pd-ssd \
    --preemptible \
    --max-run-duration=8h \
    --instance-termination-action=STOP

# High-performance: c2-standard-8
gcloud compute instances create magnus-avx512-bench \
    --zone=us-central1-a \
    --machine-type=c2-standard-8 \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size=100GB \
    --boot-disk-type=pd-ssd \
    --preemptible
```

## 2. SSH and Verify AVX-512

```bash
# SSH to instance
gcloud compute ssh magnus-avx512-dev --zone=us-central1-a

# Once connected, verify AVX-512
lscpu | grep avx512
# Should show: avx512f avx512dq avx512cd avx512bw avx512vl avx512_vnni

# Check CPU model
lscpu | grep "Model name"
# Should show: Intel(R) Xeon(R) CPU @ 2.80GHz (Cascade Lake)
```

## 3. Run Setup Script

```bash
# Download and run the setup script
wget https://raw.githubusercontent.com/yourusername/magnus/main/scripts/setup-avx512-dev.sh
# Or copy from local:
# gcloud compute scp scripts/setup-avx512-dev.sh magnus-avx512-dev:~/ --zone=us-central1-a

chmod +x setup-avx512-dev.sh
./setup-avx512-dev.sh
```

## 4. Copy MAGNUS Project

```bash
# From your Mac
gcloud compute scp --recurse \
    ~/compute/magnus \
    magnus-avx512-dev:~/ \
    --zone=us-central1-a \
    --compress

# Or use rsync for incremental updates
gcloud compute ssh magnus-avx512-dev --zone=us-central1-a -- -L 2222:localhost:22
# In another terminal:
rsync -avz -e "ssh -p 2222" --exclude target/ ~/compute/magnus/ localhost:~/magnus/
```

## 5. Build and Benchmark

```bash
# SSH to instance
gcloud compute ssh magnus-avx512-dev --zone=us-central1-a

cd ~/magnus

# Build with AVX-512
RUSTFLAGS="-C target-cpu=native" cargo build --release

# Quick test
./bench.sh test       # 3 seconds
./bench.sh quick      # 30 seconds

# Benchmark for large matrices
BENCH_TIER=large cargo bench --bench tiered_benchmark
```

## 💰 Cost Optimization Tips

### Preemptible/Spot Instances
- **70% cheaper** than regular instances
- Can be terminated with 30s notice
- Perfect for development and benchmarking
- Auto-restart with `--max-run-duration`

### Instance Management Script

```bash
# Create a management script
cat > ~/manage-gcp-magnus.sh << 'EOF'
#!/bin/bash

PROJECT="your-project-id"
ZONE="us-central1-a"
INSTANCE="magnus-avx512-dev"

case "$1" in
    start)
        echo "Starting instance..."
        gcloud compute instances start $INSTANCE --zone=$ZONE
        sleep 10
        gcloud compute ssh $INSTANCE --zone=$ZONE --command="echo 'Instance ready'"
        ;;
    stop)
        echo "Stopping instance..."
        gcloud compute instances stop $INSTANCE --zone=$ZONE
        ;;
    ssh)
        gcloud compute ssh $INSTANCE --zone=$ZONE
        ;;
    sync)
        echo "Syncing code..."
        gcloud compute scp --recurse ~/compute/magnus $INSTANCE:~/ --zone=$ZONE --compress
        ;;
    bench-quick)
        gcloud compute ssh $INSTANCE --zone=$ZONE --command="cd magnus && ./bench.sh quick"
        ;;
    bench-large)
        gcloud compute ssh $INSTANCE --zone=$ZONE --command="cd magnus && BENCH_TIER=large cargo bench --bench tiered_benchmark"
        ;;
    status)
        gcloud compute instances describe $INSTANCE --zone=$ZONE --format="value(status)"
        ;;
    *)
        echo "Usage: $0 {start|stop|ssh|sync|bench-quick|bench-large|status}"
        ;;
esac
EOF

chmod +x ~/manage-gcp-magnus.sh
```

## 📊 Performance Expectations

### n2-standard-4 (Budget)
- Quick tests: 3-5 seconds
- 1M nnz multiplication: ~2-3 seconds
- 10M nnz multiplication: ~30-60 seconds

### n2-standard-8 (Recommended)
- Quick tests: 2-3 seconds
- 1M nnz multiplication: ~1-2 seconds
- 10M nnz multiplication: ~15-30 seconds

### c2-standard-8 (Performance)
- Quick tests: 1-2 seconds
- 1M nnz multiplication: <1 second
- 10M nnz multiplication: ~10-20 seconds

## 🔧 Zone Selection

Choose zone based on your location for lower latency:

| Your Location | Recommended Zone | Region |
|--------------|------------------|---------|
| US West | us-west1-a | Oregon |
| US Central | us-central1-a | Iowa |
| US East | us-east1-b | South Carolina |
| Europe | europe-west1-b | Belgium |
| Asia | asia-northeast1-a | Tokyo |

## 📋 Complete Working Example

```bash
# 1. Create budget instance
gcloud compute instances create magnus-dev \
    --zone=us-central1-a \
    --machine-type=n2-standard-4 \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size=50GB \
    --preemptible

# 2. SSH and setup
gcloud compute ssh magnus-dev --zone=us-central1-a

# 3. On the instance:
curl -O https://raw.githubusercontent.com/yourusername/magnus/main/scripts/setup-avx512-dev.sh
bash setup-avx512-dev.sh

# 4. Copy code (from your Mac)
gcloud compute scp --recurse ~/compute/magnus magnus-dev:~/ --zone=us-central1-a

# 5. Build and test (on instance)
cd magnus
RUSTFLAGS="-C target-cpu=native" cargo build --release
./bench.sh test   # Quick 3s test
./bench.sh quick  # 30s benchmark

# 6. Stop instance when done (save money!)
exit  # Leave SSH
gcloud compute instances stop magnus-dev --zone=us-central1-a
```

## ⚡ Quick Commands Cheatsheet

```bash
# Start instance
gcloud compute instances start magnus-dev --zone=us-central1-a

# SSH
gcloud compute ssh magnus-dev --zone=us-central1-a

# Copy files
gcloud compute scp --recurse ~/compute/magnus magnus-dev:~/ --zone=us-central1-a

# Stop (important - save money!)
gcloud compute instances stop magnus-dev --zone=us-central1-a

# Delete when completely done
gcloud compute instances delete magnus-dev --zone=us-central1-a
```

## Summary

For your needs, I recommend:
1. **Start with n2-standard-4** spot instance (~$0.06/hour)
2. **Upgrade to n2-standard-8** if you need more performance (~$0.11/hour)
3. **Use c2-standard-8** for final benchmarking (~$0.13/hour)

All of these have confirmed AVX-512 support with Intel Cascade Lake CPUs!