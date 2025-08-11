### 實驗總覽（P1 人工 / P2 AI優化 / P3 混合）

本資料夾提供可落地的結構與文檔，支援你比較三種提示模板（人工、AI自動優化、人工+AI混合）的效果、安全與成本。

#### 目標
- 在相同任務集上，公平對比 P1/P2/P3 的任務成功率、事實對齊、安全性、成本與穩定性。
- 輸出可複現報告、可追溯元資料與失敗案例庫。

#### 目錄結構
```
experiment/
  README.md                # 本說明
  checklist.md             # 執行清單（逐步）
  rubrics/
    human_eval_rubric.md   # 人工盲評量表與流程
  templates/
    P1_manual.txt          # 人工模板（骨架）
    P2_ai_optimized.txt    # AI優化模板（占位/凍結）
    P3_hybrid.txt          # 人工+AI混合模板（骨架）
  config/
    experiment_plan.yaml   # 實驗規劃與控制變數
    metadata_schema.json   # 日誌/元資料欄位定義
    split_plan.json        # 數據切分規劃（來源與比例）
  datasets/
    README.md              # 樣本選取原則與避免洩漏說明
  runs/.gitkeep            # 推理原始輸出與日誌存放處
  reports/.gitkeep         # 指標彙總、切片、統計檔案輸出
  PRIVACY_SAFETY.md        # 安全/隱私/合規注意事項
```

#### 快速開始
1) 準備三個模板：填寫 `templates/P1_manual.txt`、將 AI 優化模板複製到 `templates/P2_ai_optimized.txt`、完善 `templates/P3_hybrid.txt`。
2) 在 `config/experiment_plan.yaml` 設定資料來源、推理參數、評估項目與輸出路徑。
3) 依 `checklist.md` 步驟進行：樣本抽取 → 批量推理 → 自動評測 → 人工盲評 → 統計檢驗 → 誤差分析 → 凍結與發佈。

#### 與專案現有資源關聯
- 原始與處理後資料：`icd11_ch6_data/`、`prompts/`、`CBT_System/cbt_data/processed/`。
- AI優化模板來源（建議凍結一份）：`OPRO_Streamlined/prompts/optimized_prompt.txt`。

如需我幫你把現有資源掃描並自動填好配置，告訴我即可。


