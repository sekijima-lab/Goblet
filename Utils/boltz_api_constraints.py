import os
import gc
import torch
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from pytorch_lightning import Trainer, seed_everything

# Boltz internal imports
from boltz.data.module.inferencev2 import Boltz2InferenceDataModule
from boltz.data.types import Manifest, Record
from boltz.data.mol import load_canonicals
from boltz.model.models.boltz2 import Boltz2
from boltz.main import (
    PairformerArgsV2,
    Boltz2DiffusionParams,
    MSAModuleArgs,
    BoltzSteeringParams,
    process_input,  # 前処理関数をインポート
    BoltzProcessedInput,
)
from boltz.data.write.writer import BoltzAffinityWriter, BoltzWriter
import pickle
from pathlib import Path
from typing import Any, Optional
from rdkit import Chem
import yaml

from boltz.data.types import Manifest, MSA
from boltz.data.parse.yaml import parse_yaml
from boltz.data.parse.a3m import parse_a3m
from boltz.data import const
import json

def build_1v1_manifest(
    job_id: str,
    ligand_smiles: str,
    msa_a3m_path: str,
    ccd: dict,
    mol_dir: Path,
    processed_dir: Path,
    constraints: Optional[list[dict[str, Any]]] = None,
) -> Manifest:
    """1タンパク質・1リガンド専用の超高速Manifest生成関数"""
    
    # 1. A3Mファイルからタンパク質の配列（1本目）を抽出
    protein_seq = ""
    with open(msa_a3m_path, "r") as f:
        for line in f:
            if not line.startswith(">"):
                protein_seq = line.strip()
                break
    
    if not protein_seq:
        raise ValueError("A3Mファイルからタンパク質配列を読み取れませんでした。")

    # 2. 入力を辞書として組み立て、YAMLへ安全にシリアライズする
    if constraints is not None:
        yaml_data = {
            "version": 1,
            "sequences": [
                {
                    "protein": {
                        "id": ["A"],
                        "sequence": protein_seq,
                        "msa": str(msa_a3m_path),
                    }
                },
                {"ligand": {"id": "B", "smiles": ligand_smiles}},
            ],
            "constraints": constraints,
            "properties": [{"affinity": {"binder": "B"}}],
        }
    else:
        yaml_data = {
            "version": 1,
            "sequences": [
                {
                    "protein": {
                        "id": ["A"],
                        "sequence": protein_seq,
                        "msa": str(msa_a3m_path),
                    }
                },
                {"ligand": {"id": "B", "smiles": ligand_smiles}},
            ],
            "properties": [{"affinity": {"binder": "B"}}],
        }
    print(yaml_data, flush=True)
    yaml_path = processed_dir / f"{job_id}.yaml"
    yaml_path.write_text(
        yaml.safe_dump(yaml_data, sort_keys=False),
        encoding="utf-8",
    )
    
    # 3. Boltzのパーサーでターゲット情報（構造、分子情報など）をパース
    target = parse_yaml(yaml_path, ccd, mol_dir, boltz2=True)
    #target.record.id = job_id
    #target_id = target.record.id

    # 4. A3Mファイルを読み込み、.npzキャッシュとして保存
    msa: MSA = parse_a3m(Path(msa_a3m_path), taxonomy=None, max_seqs=8192)
    msa_npz_path = processed_dir / "msa" / f"{job_id}_0.npz"
    msa.dump(msa_npz_path)

    # 5. タンパク質チェーンにMSAのIDを紐付け
    prot_id = const.chain_type_ids["PROTEIN"]
    for chain in target.record.chains:
        if chain.mol_type == prot_id:
            chain.msa_id = f"{job_id}_0"
        else:
            chain.msa_id = -1

    # 6. DataModuleが必要とするキャッシュをダンプ
    target.structure.dump(processed_dir / "structures" / f"{job_id}.npz")
    target.residue_constraints.dump(processed_dir / "constraints" / f"{job_id}.npz")
    
    Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)
    with (processed_dir / "mols" / f"{job_id}.pkl").open("wb") as f:
        pickle.dump(target.extra_mols, f)

    # 7. Manifestオブジェクトを生成して返す
    return Manifest([target.record])

app_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 サーバー起動: 構造予測モデルとAffinityモデルを両方ロードします...")
    torch.set_float32_matmul_precision("high")
    torch.set_grad_enabled(False)
    local_ = os.environ.get("HOME")
    cache_dir = Path(local_+"/.boltz")#.expanduser()
    mol_dir = cache_dir / "mols"
    conf_checkpoint = cache_dir / "boltz2_conf.ckpt"
    aff_checkpoint = cache_dir / "boltz2_aff.ckpt"
    
    # 共通パラメータ
    diffusion_params = Boltz2DiffusionParams(step_scale=1.5)
    pairformer_args = PairformerArgsV2()
    msa_args = MSAModuleArgs(subsample_msa=True, num_subsampled_msa=1024, use_paired_feature=True)

    # 1. 構造予測モデル (Conf) のロード
    steering_args_conf = BoltzSteeringParams(fk_steering=False, physical_guidance_update=False, contact_guidance_update=True)
    predict_args_conf = {
        "recycling_steps": 3, "sampling_steps": 200, "diffusion_samples": 1,
        "max_parallel_samples": 5, "write_confidence_summary": False, "write_full_pae": False, "write_full_pde": False,
    }
    model_conf = Boltz2.load_from_checkpoint(
        conf_checkpoint, strict=True, predict_args=predict_args_conf, map_location="cpu",
        diffusion_process_args=asdict(diffusion_params), ema=False,
        pairformer_args=asdict(pairformer_args), msa_args=asdict(msa_args), steering_args=asdict(steering_args_conf),
    )
    model_conf.eval()

    # 2. Affinityモデル (Aff) のロード
    steering_args_aff = BoltzSteeringParams(fk_steering=False, physical_guidance_update=False, contact_guidance_update=False)
    predict_args_aff = {
        "recycling_steps": 5, "sampling_steps": 200, "diffusion_samples": 3,
        "max_parallel_samples": 1, "write_confidence_summary": False, "write_full_pae": False, "write_full_pde": False,
    }
    model_aff = Boltz2.load_from_checkpoint(
        aff_checkpoint, strict=True, predict_args=predict_args_aff, map_location="cpu",
        diffusion_process_args=asdict(diffusion_params), ema=False,
        pairformer_args=asdict(pairformer_args), msa_args=asdict(msa_args), steering_args=asdict(steering_args_aff),
        affinity_mw_correction=False,
    )
    model_aff.eval()
    
    trainer = Trainer(
        accelerator="gpu", devices=1, precision="bf16-mixed",
        logger=False, enable_progress_bar=False, enable_model_summary=False
    )
    
    app_state["model_conf"] = model_conf
    app_state["model_aff"] = model_aff
    app_state["trainer"] = trainer
    app_state["ccd"] = load_canonicals(mol_dir)
    app_state["mol_dir"] = mol_dir
    
    print("✅ 2つのモデルの準備完了！")
    yield
    app_state.clear()

app = FastAPI(lifespan=lifespan)

# ---------------------------------------------------------
# 3. 推論エンドポイント
# ---------------------------------------------------------
# APIリクエストには本当にこの2つ（とID）だけを要求します
class DockingRequest(BaseModel):
    ligand_smiles: str
    msa_a3m_path: str
    job_id: str
    constraints: Optional[list[dict[str, Any]]] = None

from boltz.data.write.writer import BoltzWriter, BoltzAffinityWriter # BoltzWriter もインポートしてください
import time
@app.post("/predict_affinity")
async def predict_affinity(req: DockingRequest):
    __start = time.time()
    seed_everything(42, workers=True)
    t4_tmp_env = os.environ.get("T4TMPDIR")
    base_tmp_dir = Path(t4_tmp_env+"/wkdir").resolve()
    base_tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"boltz_{req.job_id}_", dir=base_tmp_dir))
    
    try:
        model_conf = app_state["model_conf"]
        model_aff = app_state["model_aff"]
        trainer = app_state["trainer"]
        ccd = app_state["ccd"]
        mol_dir = app_state["mol_dir"]
        
        # ディレクトリ準備等は前回と同じ
        processed_dir = tmp_dir / "processed"
        out_dir = tmp_dir / "predictions"
        for d in [processed_dir / "msa", processed_dir / "structures", 
                  processed_dir / "constraints", processed_dir / "mols", out_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # 超軽量Manifest関数の呼び出し
        manifest = build_1v1_manifest(
            job_id=req.job_id, ligand_smiles=req.ligand_smiles, msa_a3m_path=req.msa_a3m_path,
            ccd=ccd, mol_dir=mol_dir, processed_dir=processed_dir,
            constraints=req.constraints,
        )
        
        # ==========================================
        # STEP 1: 構造予測の実行 (ここで pre_affinity_xxx.npz が出力される)
        # ==========================================
        conf_writer = BoltzWriter(
            data_dir=processed_dir / "structures", output_dir=out_dir, 
            output_format="mmcif", boltz2=True, write_embeddings=False
        )
        trainer.callbacks = [conf_writer]
        
        data_module_conf = Boltz2InferenceDataModule(
            manifest=manifest, target_dir=processed_dir / "structures", msa_dir=processed_dir / "msa",
            mol_dir=mol_dir, num_workers=1, constraints_dir=processed_dir / "constraints",
            extra_mols_dir=processed_dir / "mols", override_method="other"
        )
        trainer.predict(model_conf, datamodule=data_module_conf, return_predictions=False)

        # ==========================================
        # STEP 2: Affinity予測の実行 (先ほど出力された npz を読み込む)
        # ==========================================
        aff_writer = BoltzAffinityWriter(data_dir=processed_dir / "structures", output_dir=out_dir)
        trainer.callbacks = [aff_writer]
        
        data_module_aff = Boltz2InferenceDataModule(
            manifest=manifest, 
            target_dir=out_dir, # ★ここ重要: STEP1の出力先を指定する
            msa_dir=processed_dir / "msa",
            mol_dir=mol_dir, num_workers=1, constraints_dir=processed_dir / "constraints",
            extra_mols_dir=processed_dir / "mols", override_method="other", affinity=True
        )
        trainer.predict(model_aff, datamodule=data_module_aff, return_predictions=False)
        
        result_json_path = out_dir / req.job_id / f"affinity_{req.job_id}.json"
        
        affinity_score = None
        
        if result_json_path.exists():
            with open(result_json_path, "r") as f:
                result_data = json.load(f)
                # JSONから "affinity_pred_value" の数値をピンポイントで取得
                affinity_score = result_data.get("affinity_pred_value")
        else:
            raise FileNotFoundError(f"結果ファイルが見つかりません: {result_json_path}")
            
        return {
            "status": "success", 
            "job_id": req.job_id,
            "affinity_score": affinity_score
        }
    except ValueError as e:
        # Boltz (RDKit) がSMILESの立体構築に失敗した場合など、入力値に起因するエラー
        error_msg = str(e)
        print(f"⚠️ [Client Error] {error_msg}")
        raise HTTPException(
            status_code=422, # または 400
            detail=f"Invalid SMILES or computation failed: {error_msg}"
        ) from e
    finally:
        __end = time.time()
        print(f"Boltz: consumed {__end - __start} [s]")
        shutil.rmtree(tmp_dir, ignore_errors=True)# 2. 不要になった変数を明示的に削除（DataModuleなどをメモリから消す）
        if 'data_module_conf' in locals(): del data_module_conf
        if 'data_module_aff' in locals(): del data_module_aff
        if 'manifest' in locals(): del manifest
        
        # 3. Pythonのガベージコレクションを強制実行（CPUメモリの解放）
        gc.collect()
        
        # 4. PyTorchのVRAMキャッシュを完全に空にする（GPUメモリの解放）
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # デバッグ用：解放後のVRAM使用状況を確認したい場合は以下のコメントを外す
        print(f"🧹 [Memory Cleared] Allocated: {torch.cuda.memory_allocated()/1024**3:.1f}GB, Reserved: {torch.cuda.memory_reserved()/1024**3:.1f}GB")
        #pass
if __name__ == "__main__":
    # 環境変数 "API_PORT" があればそれを使い、なければデフォルトの 8000 を使う
    port = int(os.environ.get("API_PORT", 8000))
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")