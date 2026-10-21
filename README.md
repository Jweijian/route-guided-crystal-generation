# Route-guided crystal generation

当前版本只包含这一条流程：**ELM 化学式生成 → 本地 RAG / DeepSeek 合成路线与英文结构描述 → CrystalFormer-CSP 晶体候选生成**。

仓库包含代码、Notebook、配置、提示词、JSON Schema、测试、第三方许可，以及当前流程所需的 ELM / CrystalFormer 权重和合成数据库。运行资源通过 Git LFS 保存，缓存与历史结果不上传。

## 获取完整仓库

先安装 [Git LFS](https://git-lfs.com/)，再克隆；不要只下载源码 ZIP，因为其中可能只有大文件指针。

```powershell
git lfs install
git clone https://github.com/Jweijian/route-guided-crystal-generation.git
cd route-guided-crystal-generation
git lfs pull
```

仓库目前为私有，下载者需要获得仓库访问权限。

## 安装

使用 Python 3.10；Windows 的 CrystalFormer 使用 JAX CPU。已有 `highk_bec` 环境也可使用。

```powershell
conda create -n route-crystal python=3.10 -y
conda activate route-crystal
python -m pip install -r requirements.txt
python -m ipykernel install --user --name route-crystal --display-name "Python (route-crystal)"
```

默认 ELM 使用 CPU；有 NVIDIA GPU 时，可安装适配的 PyTorch CUDA wheel，并将 `crystal_pipeline/config.json` 的 `elm.device` 改为 `cuda`。CrystalFormer 沿用已验证的 JAX CPU 配置。

## 验证运行资源

`git lfs pull` 会把运行文件恢复到下面的位置，`runtime_assets.json` 记录文件大小和 SHA256：

| 文件 | 放置位置 |
| --- | --- |
| ELM ionic 权重 | `crystal_pipeline/assets/elm/gpt2_chem_ionic.pth` |
| ELM alloy 权重（仅 alloy 模式需要） | `crystal_pipeline/assets/elm/gpt2_chem_alloy.pth` |
| 合成数据库 | `crystal_pipeline/assets/database/all_merged_slimmed.json` |
| CrystalFormer 权重 | `vendor/CrystalFormer-CSP/checkpoints/alex20s/csp/epoch_046000.pkl` |

这些文件均已随仓库上传，无需另行提供数据库或手工复制权重。安装依赖后运行完整性检查：

```powershell
python scripts/check_runtime_assets.py
```

应显示 `Verified 4 runtime files.`；若提示大小错误，请确认已安装 Git LFS 并重新运行 `git lfs pull`。CrystalFormer checkpoint 的上游下载链接仍记录在清单中。

## API 设置

路线和描述阶段使用 DeepSeek。通过环境变量设置密钥，代码中不保存密钥：

```powershell
$env:DEEPSEEK_API_KEY = "你的密钥"
```

`config.json` 的 `api` 节控制模型和推理强度；当前为 `deepseek-v4-pro`、`low`。`routes.formula_workers=4` 控制化学式并发数。第二阶段结构生成不调用 API。

## 运行完整流程

在项目根目录启动 `jupyter lab`，打开 `crystal_pipeline/运行完整流程.ipynb`，选择已安装依赖的内核，按顺序运行。Notebook 先保存描述，再调用 CrystalFormer。

命令行也可分两阶段运行：

```powershell
python crystal_pipeline/run.py --stage descriptions
python crystal_pipeline/run.py --stage crystals --resume-run "第一阶段打印的批次目录"
```

直接运行全部阶段：

```powershell
python crystal_pipeline/run.py
```

`pipeline.formula_source="elm"` 使用 ELM；改为 `file` 可读取 `crystal_pipeline/inputs/formulas.json`。数量由 `elm.num_formulas` 控制。第二阶段默认每个结构假设原始采样 8 个，参数位于 `crystalformer` 节。

单独指定化学式和空间群生成结构，可打开项目根目录的 `CrystalFormer_固定化学式生成.ipynb`。

## 输出与生成条件

结果保存在 `crystal_pipeline/outputs/<批次>/`。每个结构假设取目标化学式及首个 `space_group_candidates` 编号，调用真实预训练 CrystalFormer 生成晶格、Wyckoff 占位与坐标，不手写晶格常数。

每个假设的原始 CSV、组成筛选 CSV、日志、全部通过检查的 `candidate_*.cif` 与元数据保存在 `crystals_crystalformer/Fxxx/Rxx/Hxx/`。第一份通过检查的结构另存为 `candidate.cif`，供报告使用；不按能量排序。采样数量不等于通过筛选的数量，没有通过检查的样本时记录失败并继续其他假设。

**模型只接收化学式和空间群。路线、配位和拓扑描述会保存，但没有直接输入 CrystalFormer。** 输出为未弛豫候选，通过组成、实际空间群和原子重叠检查不代表满足全部描述或已经稳定。空间群检查使用 `symprec=0.01 Å`、`angle_tolerance=5°`。

## 测试

以下离线测试不需要 API 密钥、模型权重或合成数据库：

```powershell
python -m unittest discover -s crystal_pipeline/verification -p "test_*.py"
```

使用仓库里的真实权重和数据库验证 ELM、RAG、CrystalFormer，且不请求 API：

```powershell
python scripts/smoke_test.py
```

烟雾测试会生成一个 ELM 化学式、检索数据库，并为 `LiCu2OF3 / 空间群14` 采样结构。完整流程的路线与描述阶段仍需要用户自己的 DeepSeek API 密钥。

第三方源码与来源见 `THIRD_PARTY_NOTICES.md`。仓库中的算法整合保持当前流程，未包含其他历史结构生成后端。
