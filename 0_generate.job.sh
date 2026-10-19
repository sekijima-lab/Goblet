#!/bin/bash
#$ -cwd
#$ -l node_q=1
#$ -l h_rt=24:00:00
#$ -t 2:2
#$ -m eas
#$ -N ampc
# . /etc/profile.d/modules.sh
# akt1 ampc cxcr4 cp3a4 hivrt hivpr gcr kif11
export BOLTZ_CACHE_DIR="$T4TMPDIR/boltz_cache"
mkdir ${T4TMPDIR}/boltz_cache
cp -ar $HOME/.boltz/. $BOLTZ_CACHE_DIR  &

module load cuda/12.1.0 gcc miniconda/24.1.2
source ~/.bashrc
eval "$(/apps/t4/rhel9/free/miniconda/24.1.2/bin/conda shell.bash hook)"

export HYDRA_FULL_ERROR=1
fg
conda activate boltz
# 1. OSに問い合わせて、現在空いているランダムなポート番号を自動取得する（HPCの鉄則）
FREE_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
echo "🔍 空きポート ${FREE_PORT} 番を確保しました。"
export API_PORT=$FREE_PORT
python -u Utils/boltz_api_constraints.py > boltz_api_${API_PORT}.log 2>&1 &
API_PID=$!
trap "echo '🛑 探索終了。APIサーバー(PID: $API_PID)をシャットダウンし、VRAMを解放します...'; kill $API_PID" EXIT
echo "⏳ モデルのロードとサーバーの立ち上がりを待機しています..."

# サーバー（ポート8000）が応答を返すまで、2秒おきに無限ループでノックし続ける
until curl -s http://127.0.0.1:$API_PORT/docs > /dev/null; do
    sleep 2
done

echo "✅ サーバーの準備が完了しました！"

conda activate goblet

python ./Generator/mcts.py mcts.init_smiles='"Cc1cccc(NC(CS)=NN)c1"' mcts.reward_name="ASBoltz" mcts.output_fname="${JOB_NAME}_$SGE_TASK_ID.ASBOLTZ.csv" mcts.work_space="/wkdir/${JOB_NAME}_$SGE_TASK_ID/"

#echo $SGE_TASK_ID
conda deactivate
#deactivate
echo finish
date
