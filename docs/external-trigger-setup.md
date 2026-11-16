# 美股推播：外部準時觸發設定（cron-job.org）

目的：用外部服務準時呼叫 GitHub，立刻執行 us_push.yml；GitHub 內建 cron 保留當備援。
前提：本 PR 已合併到 main。若台股那份已設好，帳號與 token 做法相同，步驟 1、2 可略（token 需另加 pro-terminal 權限，見下）。

## 步驟 1：GitHub token（只給這個用途）
1. GitHub → Settings → Developer settings → Fine-grained tokens → Generate new token
2. Repository access 選 `pro-terminal`（可與台股共用同一支，改選兩個 repo）
3. Permissions → **Actions: Read and write**（其他都不要）
4. 複製 token（只顯示一次，不要貼到聊天）

## 步驟 2：建立三個排程（共用設定）
- URL：`https://api.github.com/repos/Joesung0719/pro-terminal/actions/workflows/us_push.yml/dispatches`
- Request method：POST
- Headers：`Authorization: Bearer <token>`、`Accept: application/vnd.github+json`、`X-GitHub-Api-Version: 2022-11-28`、`Content-Type: application/json`、`User-Agent: cron-job-org`
- **Time zone：America/New_York**（夏令/冬令由 cron-job.org 自動處理，只需建三個，不用像 GitHub cron 排兩個 UTC）
- Schedule：週一至週五；預期回應 HTTP 204

| 排程 | 美東時間 | Body |
|---|---|---|
| 開盤半小時 | 10:05 | `{"ref":"main","inputs":{"cmd":"auto","slot":"stage1"}}` |
| 午盤 | 12:35 | `{"ref":"main","inputs":{"cmd":"auto","slot":"stage2"}}` |
| 收盤 | 16:20 | `{"ref":"main","inputs":{"cmd":"auto","slot":"stage3"}}` |

注意：Body 的 `cmd` 一定要是 `auto`（預設是 test，只會送測試訊息）。

## 步驟 3：驗證
1. cron-job.org 對任一排程按 Test run → 回 204
2. GitHub → Actions 會出現 `US LINE stage1`（或對應時段）
3. 同時段後來由內建 cron 備援觸發的執行會顯示「已取消」，LINE 不會重複收到
4. 非交易時間測試會因「過早／過時」被跳過（日誌有 ⏭️ 說明）

## 延遲保護規則
- 執行時間比預定（美東 10:05／12:35／16:20）晚超過 120 分，或早超過 10 分 → 不推
- 晚 15 分以上、120 分以內 → 照推，標題時間後加「（排程延遲 N 分）」
- 時段由 cron 字串／slot 決定，不看執行當下的小時，延遲不會被判成別的時段
- 休市日（週末、NYSE 假日）沿用原本日曆判斷，不推播
