# Function-Calling Fine-Tune PoC on Nebius

End-to-end recipe for fine-tuning a 72B-parameter model on 16 H200 GPUs, on
a SLURM cluster deployed with [Soperator](https://github.com/nebius/soperator).

## What's in this repo

```
davide_demoday
├── README.md                
├── terraform/                    ← Provisions the SLURM cluster on Nebius
│   ├── main.tf
│   ├── variables.tf
│   └── terraform.tfvars.example
└── qwen72b_finetuning/           ← Fine-tuning example
    ├── README.md            
    ├── setup_environment.sh
    ├── prepare_data.sh
    ├── launch.sbatch
    └── scripts/
```

## Prerequisites

### Accounts & access

- **Nebius account**
  - **16 H200 GPUs (gpu-h200-sxm)** quota in your Nebius project.
  - **1 TB shared file system** quota
  - **1 public IP** quota
- **Hugging Face account** with a read token. Generate at https://huggingface.co/settings/tokens.
- **Accepted terms** for the gated  dataset. One-click at
      https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2 —
      without this, data preparation fails.

### Local tools

- **Terraform** ≥ 1.6
- **Nebius CLI**
- **kubectl**
- **jq**
- **coreutils**
- **SSH key pair**

## Infrastructure deployment

This project comes with a pre-configured Soperator installation directory
located at `davide_demoday/soperator/installations/acmepoc`.

### 1. Clone this repo

```bash
git clone https://github.com/vanzod/davide_demoday.git
cd davide_demoday
```

### 2. Environment setup

Set your `NEBIUS_TENANT_ID` and `NEBIUS_PROJECT_ID` in the `.envrc` file, then run:

```bash
cd soperator/installations/acmepoc
source .envrc
```

Check that nebius CLI is authenticated:

```bash
nebius iam whoami
```

Add your SSH public key (assuming stored at `~/.ssh/id_ed25519.pub) to the terraform
variables:

```bash
sed -i "s|<SSH_PUBLIC_KEY>|$(cat ~/.ssh/id_ed25519.pub)|g" terraform.tfvars
```

### 3. Create shared file systems

Create two jail file systems to persist shared and data directories
across different clusters deployments.

```bash
nebius compute filesystem create \
  --name acme-shared \
  --size-gibibytes 100 \
  --type network_ssd \
  --block-size-bytes 4096

nebius compute filesystem create \
  --name acme-data \
  --size-gibibytes 800 \
  --type network_ssd \
  --block-size-bytes 4096
```

Then retrieve the file systems IDs with:

```bash
SHARED_ID=$(nebius compute filesystem get-by-name --name acme-shared --format json | jq -r ".metadata.id")
DATA_ID=$(nebius compute filesystem get-by-name --name acme-data --format json | jq -r ".metadata.id")
```

and add them in the respective fields in the terraform variables:

```bash
sed -i "s|<JAIL_SHARED_ID>|${SHARED_ID}|g" terraform.tfvars
sed -i "s|<JAIL_DATA_ID>|${DATA_ID}|g" terraform.tfvars
```

### 4. Deploy the Slurm cluster

```bash
terraform init
terraform apply
```

### 5. Connect to the login node

```bash
export SLURM_IP=$(terraform state show module.login_script.terraform_data.lb_service_ip | grep 'input' | grep -oE '[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+' | head -n 1)
ssh root@$SLURM_IP
```

Welcome to your Slurm cluster!

### 6. Clone this repsitory on the cluster

```bash
# On the Slurm cluster login node
git clone https://github.com/vanzod/davide_demoday.git
```

### 7. Run the fine-tuning example

```bash
cd davide_demoday/qwen72b_finetuning
export HF_TOKEN=hf_xxxxxxxxxxxx           # your Hugging Face token

./setup_environment.sh
./prepare_data.sh
sbatch launch.sbatch
```

For detailed output descriptions, metric explanations, expected numbers, and
troubleshooting of the training itself, see
[`qwen72b_finetuning/README.md`](qwen72b_finetuning/README.md).

### 9. Tear down the cluster

On your local machine:

```bash
cd soperator/installations/acmepoc
source .envrc
terraform destroy
```
