# #!/bin/bash
# export USE_LIBUV=0


# now=$(date +"%Y%m%d_%H%M%S")

# config=configs/glas_10.yaml
# labeled_id_path=partitions/glas_10/labeled.txt
# unlabeled_id_path=partitions/glas_10/unlabeled.txt
# save_path=exp/glas/10/corrmatch

# mkdir -p $save_path

# torchrun --nproc_per_node=1 --master_port=29500 UniSemAlign.py `
#   --config=configs/glas_10.yaml `
#   --labeled-id-path partitions/glas_10/labeled.txt `
#   --unlabeled-id-path partitions/glas_10/unlabeled.txt `
#   --save-path exp/glas/10/corrmatch `
#   --port 29500

#!/bin/bash
set -e

export USE_LIBUV=0
now=$(date +"%Y%m%d_%H%M%S")

config=${1:-configs/glas_10.yaml}
labeled_id_path=${2:-partitions/glas_10/labeled.txt}
unlabeled_id_path=${3:-partitions/glas_10/unlabeled.txt}
save_path=${4:-exp/glas/10/corrmatch}
port=${5:-29500}

mkdir -p "$save_path"

USE_LIBUV=0 torchrun --nproc_per_node=1 --master_port=$port UniSemAlign.py \
  --config="$config" \
  --labeled-id-path "$labeled_id_path" \
  --unlabeled-id-path "$unlabeled_id_path" \
  --save-path "$save_path" \
  --port $port \
  2>&1 | tee "$save_path/$now.txt"
